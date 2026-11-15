-- Create or replace table for tracking uploaded documents
CREATE OR REPLACE TABLE doc_ai_qs_db.doc_ai_schema.UDM_TRACKER (
    file_name STRING,
    file_size NUMBER,
    last_modified TIMESTAMP,
    snowflake_file_url STRING,
    processed BOOLEAN DEFAULT FALSE
);

-- Create or replace table for extracting document data
CREATE OR REPLACE TABLE doc_ai_qs_db.doc_ai_schema.UDM_EXTRACTS (
    FILENAME STRING,
    ocrScore FLOAT,
    TOTAL_DEBT_1 STRING,
    TOTAL_DEBT_1_score FLOAT,
    TOTAL_DEBT_2 STRING,
    TOTAL_DEBT_2_score FLOAT,
    CURRENT_ASSET STRING,
    CURRENT_ASSET_score FLOAT,
    FIXED_ASSET STRING,
    FIXED_ASSET_score FLOAT,
    Companyname STRING,
    Companyname_score FLOAT,
    CHN STRING,
    CHN_score FLOAT,
    REPORTING_DATE STRING,
    REPORTING_DATE_score FLOAT,
    EQUITY STRING,
    EQUITY_score FLOAT,
    REVENUE STRING,
    REVENUE_score FLOAT,
    FILE_SIZE INTEGER,
    LAST_MODIFIED TIMESTAMP,
    URL STRING
);

-- Create or replace task for processing documents
CREATE OR REPLACE TASK doc_ai_qs_db.doc_ai_schema.UDM_TASK
    WAREHOUSE = DOC_AI_QS_WH
    SCHEDULE = 'USING CRON */5 * * * * Asia/Kolkata'
AS
BEGIN
    -- Insert new files into the tracker table
    INSERT INTO doc_ai_qs_db.doc_ai_schema.UDM_TRACKER (file_name, file_size, last_modified, snowflake_file_url, processed)
    SELECT 
        relative_path AS file_name,
        size AS file_size,
        last_modified::TIMESTAMP_NTZ AS last_modified,
        file_url AS snowflake_file_url,
        FALSE AS processed
    FROM directory(@doc_ai_stage) AS stage_files
    WHERE NOT EXISTS (
        SELECT 1 
        FROM doc_ai_qs_db.doc_ai_schema.UDM_TRACKER AS tracker
        WHERE stage_files.relative_path = tracker.file_name
    );

    -- Select files to process and insert data into the extracts table
    INSERT INTO doc_ai_qs_db.doc_ai_schema.UDM_EXTRACTS
    SELECT
        file_name AS FILENAME,
        json:__documentMetadata.ocrScore::FLOAT AS ocrScore,
        json:Creditor1[0]:value::STRING AS TOTAL_DEBT_1,
        json:Creditor1[0]:score::FLOAT AS TOTAL_DEBT_1_score,
        json:Creditor2[0]:value::STRING AS TOTAL_DEBT_2,
        json:Creditor2[0]:score::FLOAT AS TOTAL_DEBT_2_score,
        json:CurrentAsset[0]:value::STRING AS CURRENT_ASSET,
        json:CurrentAsset[0]:score::FLOAT AS CURRENT_ASSET_score,
        json:FixedAsset[0]:value::STRING AS FIXED_ASSET,
        json:FixedAsset[0]:score::FLOAT AS FIXED_ASSET_score,
        json:Companyname[0]:value::STRING AS Companyname,
        json:Companyname[0]:score::FLOAT AS Companyname_score,
        json:Registrationnumber[0]:value::STRING AS CHN,
        json:Registrationnumber[0]:score::FLOAT AS CHN_score,
        json:ReportingDate[0]:value::STRING AS REPORTING_DATE,
        json:ReportingDate[0]:score::FLOAT AS REPORTING_DATE_score,
        json:Equity[0]:value::STRING AS Equity,
        json:Equity[0]:score::FLOAT AS Equity_score,
        json:Revenue[0]:value::STRING AS REVENUE,
        json:Revenue[0]:score::FLOAT AS REVENUE_score,
        file_size AS FILE_SIZE,
        last_modified AS LAST_MODIFIED,
        snowflake_file_url AS URL
    FROM (
        SELECT 
            file_name,
            file_size,
            last_modified,
            snowflake_file_url,
            DOC_AI_QS_DB.DOC_AI_SCHEMA.UDM!PREDICT(get_presigned_url('@doc_ai_stage', file_name), 1) AS json
        FROM doc_ai_qs_db.doc_ai_schema.UDM_TRACKER
        WHERE processed = FALSE
        LIMIT 20
    );

    -- Mark the processed files
    UPDATE doc_ai_qs_db.doc_ai_schema.UDM_TRACKER
    SET processed = TRUE
    WHERE file_name IN (
        SELECT file_name 
        FROM doc_ai_qs_db.doc_ai_schema.UDM_TRACKER
        WHERE processed = FALSE
        LIMIT 20
    );
END;

-- Execute Task
ALTER TASK doc_ai_qs_db.doc_ai_schema.UDM_TASK RESUME;
