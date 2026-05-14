from flask import Flask, render_template, request, jsonify
from google.cloud import storage, bigquery
import pandas as pd
import io
import json
from datetime import datetime, timezone
import google.generativeai as genai

app = Flask(__name__)

PROJECT_ID = "famous-gearing-496214-i7"
BUCKET_NAME = "sales-analytics-raw-data"
DATASET_ID = "sales_analytics"

# Clients initialisés une seule fois
storage_client = storage.Client()
bq_client = bigquery.Client()

genai.configure(api_key="AIzaSyC09PA7RdJNd6GXSbXykCRz9Kt90kyh5sY")
gemini = genai.GenerativeModel("gemini-2.5-flash")


def init_metadata_table():
    schema = [
        bigquery.SchemaField("model_name", "STRING"),
        bigquery.SchemaField("table_name", "STRING"),
        bigquery.SchemaField("target_col", "STRING"),
        bigquery.SchemaField("feature_cols", "STRING"),
        bigquery.SchemaField("model_type", "STRING"),
        bigquery.SchemaField("created_at", "TIMESTAMP"),
    ]
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.model_metadata"
    table = bigquery.Table(table_ref, schema=schema)
    bq_client.create_table(table, exists_ok=True)


def format_value_for_sql(v):
    try:
        float(v)
        return str(v)
    except (ValueError, TypeError):
        return f"'{v}'"


# ── Routes de base ────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/tables')
def get_tables():
    tables = bq_client.list_tables(f"{PROJECT_ID}.{DATASET_ID}")
    return jsonify([t.table_id for t in tables if t.table_id != 'model_metadata'])


@app.route('/columns/<table_name>')
def get_columns(table_name):
    table = bq_client.get_table(f"{PROJECT_ID}.{DATASET_ID}.{table_name}")
    return jsonify([field.name for field in table.schema])


# ── Upload ────────────────────────────────────────────────────────────────────

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    file_content = file.read()
    table_name = file.filename.replace('.csv', '').replace('-', '_')

    # 1. Upload dans Cloud Storage
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(file.filename)
    blob.upload_from_string(file_content)

    # 2. Lit le CSV
    try:
        df = pd.read_csv(io.BytesIO(file_content), encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(file_content), encoding='latin-1')

    # 3. Load job (batch, gratuit)
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        autodetect=True,
    )
    job = bq_client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()

    return jsonify({
        'success': f'{len(df)} rows loaded into table "{table_name}" successfully',
        'columns': list(df.columns),
        'table': table_name
    })


# ── Analytics ─────────────────────────────────────────────────────────────────

@app.route('/analytics/<table_name>')
def analytics(table_name):
    query = f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{table_name}` LIMIT 100"
    results = bq_client.query(query).to_dataframe()
    return jsonify(results.astype(str).to_dict(orient='records'))


# ── Gemini — Ask AI (text-to-SQL) ─────────────────────────────────────────────

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data['question']
    table_name = data['table']

    table = bq_client.get_table(f"{PROJECT_ID}.{DATASET_ID}.{table_name}")
    schema_str = ', '.join([f"{f.name} ({f.field_type})" for f in table.schema])

    prompt = f"""
You are a BigQuery SQL expert.
Table: `{PROJECT_ID}.{DATASET_ID}.{table_name}`
Schema: {schema_str}
Question: {question}

Return ONLY a SQL SELECT query. No explanation, no markdown, no backticks.
Use BigQuery syntax. Max 100 rows.
"""
    sql = gemini.generate_content(prompt).text.strip()

    if not sql.upper().startswith('SELECT'):
        return jsonify({'error': 'Only SELECT queries are allowed'}), 400

    results = bq_client.query(sql).to_dataframe()
    return jsonify({'sql': sql, 'data': results.astype(str).to_dict(orient='records')})


# ── Gemini — Auto Insights ────────────────────────────────────────────────────

@app.route('/insights/<table_name>')
def insights(table_name):
    table = bq_client.get_table(f"{PROJECT_ID}.{DATASET_ID}.{table_name}")
    schema_str = ', '.join([f"{f.name} ({f.field_type})" for f in table.schema])

    sample = bq_client.query(
        f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{table_name}` LIMIT 10"
    ).to_dataframe().astype(str).to_dict(orient='records')

    prompt = f"""
You are a business analyst. Analyze this dataset and give 3-5 key business insights.
Table: {table_name}
Schema: {schema_str}
Sample data: {json.dumps(sample)}
Be concise and focus on actionable business value.
"""
    response = gemini.generate_content(prompt)
    return jsonify({'insights': response.text})


# ── BigQuery ML — Train ───────────────────────────────────────────────────────

@app.route('/train', methods=['POST'])
def train_model():
    data = request.json
    table_name = data['table']
    target_col = data['target']
    feature_cols = data['features']
    model_type = data.get('model_type', 'linear_reg')
    model_name = f"{table_name}_model"

    features_sql = ', '.join(feature_cols)

    query = f"""
        CREATE OR REPLACE MODEL `{PROJECT_ID}.{DATASET_ID}.{model_name}`
        OPTIONS(
            model_type='{model_type}',
            input_label_cols=['{target_col}']
        ) AS
        SELECT {features_sql}, {target_col}
        FROM `{PROJECT_ID}.{DATASET_ID}.{table_name}`
        WHERE {target_col} IS NOT NULL
        AND {target_col} != ''
    """

    bq_client.query(query).result()

    # Supprime l'ancienne entrée metadata
    bq_client.query(f"""
        DELETE FROM `{PROJECT_ID}.{DATASET_ID}.model_metadata`
        WHERE model_name = '{model_name}'
    """).result()

    # Sauvegarde les features, target et model_type
    bq_client.query(f"""
        INSERT INTO `{PROJECT_ID}.{DATASET_ID}.model_metadata`
        (model_name, table_name, target_col, feature_cols, model_type, created_at)
        VALUES (
            '{model_name}',
            '{table_name}',
            '{target_col}',
            '{json.dumps(feature_cols)}',
            '{model_type}',
            CURRENT_TIMESTAMP()
        )
    """).result()

    return jsonify({'success': f'Model {model_name} ({model_type}) trained successfully'})


# ── BigQuery ML — Model features ──────────────────────────────────────────────

@app.route('/model_features/<table_name>')
def get_model_features(table_name):
    model_name = f"{table_name}_model"

    results = bq_client.query(f"""
        SELECT target_col, feature_cols
        FROM `{PROJECT_ID}.{DATASET_ID}.model_metadata`
        WHERE model_name = '{model_name}'
        ORDER BY created_at DESC LIMIT 1
    """).to_dataframe()

    if results.empty:
        return jsonify({'error': 'No model found. Train a model first.'}), 404

    return jsonify({
        'target': results.iloc[0]['target_col'],
        'features': json.loads(results.iloc[0]['feature_cols']),
        'model_type': 'unknown'
    })


# ── BigQuery ML — Predict (manuel) ───────────────────────────────────────────

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    table_name = data['table']
    model_name = f"{table_name}_model"
    values = data['values']

    values_sql = ', '.join([
        f"{format_value_for_sql(v)} AS {k}" for k, v in values.items()
    ])

    results = bq_client.query(f"""
        SELECT * FROM ML.PREDICT(
            MODEL `{PROJECT_ID}.{DATASET_ID}.{model_name}`,
            (SELECT {values_sql})
        )
    """).to_dataframe()

    return jsonify(results.astype(str).to_dict(orient='records'))


# ── Gemini + BigQuery ML — Predict en langage naturel ────────────────────────

@app.route('/predict_nl', methods=['POST'])
def predict_nl():
    data = request.json
    table_name = data['table']
    sentence = data['sentence']
    model_name = f"{table_name}_model"

    results = bq_client.query(f"""
        SELECT target_col, feature_cols
        FROM `{PROJECT_ID}.{DATASET_ID}.model_metadata`
        WHERE model_name = '{model_name}'
        ORDER BY created_at DESC LIMIT 1
    """).to_dataframe()

    if results.empty:
        return jsonify({'error': 'No trained model found. Train a model first.'}), 404

    features = json.loads(results.iloc[0]['feature_cols'])
    target = results.iloc[0]['target_col']

    prompt = f"""
Extract feature values from this sentence and return ONLY a JSON object.
Features to extract: {features}
Sentence: "{sentence}"

Rules:
- Return ONLY valid JSON, no explanation, no markdown, no backticks
- Use null if a feature is not mentioned
- Example: {{"position": "Manager", "city": "Paris", "performance_score": 85}}
"""
    raw = gemini.generate_content(prompt).text.strip().replace('```json', '').replace('```', '')
    values = json.loads(raw)
    values = {k: v for k, v in values.items() if v is not None}

    if not values:
        return jsonify({'error': 'Could not extract any feature values from your sentence.'}), 400

    values_sql = ', '.join([f"{format_value_for_sql(v)} AS {k}" for k, v in values.items()])

    pred_result = bq_client.query(f"""
        SELECT * FROM ML.PREDICT(
            MODEL `{PROJECT_ID}.{DATASET_ID}.{model_name}`,
            (SELECT {values_sql})
        )
    """).to_dataframe().astype(str).to_dict(orient='records')[0]

    predicted_val = next((v for k, v in pred_result.items() if k.startswith('predicted')), None)

    return jsonify({
        'extracted_values': values,
        'target': target,
        'prediction': predicted_val
    })


if __name__ == '__main__':
    init_metadata_table()
    app.run(host='0.0.0.0', port=8080, debug=False)
