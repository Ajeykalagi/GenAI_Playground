-- Raise Role Access
USE ROLE ACCOUNTADMIN;

-- --------------------------------------------
-- --------- Create Role and Grant Access -----
-- --------------------------------------------

-- Create a DOC AI role to be used for the quickstart
CREATE ROLE doc_ai_qs_role;
GRANT DATABASE ROLE SNOWFLAKE.DOCUMENT_INTELLIGENCE_CREATOR TO ROLE doc_ai_qs_role;

-- Grant the role to a user
GRANT ROLE doc_ai_qs_role TO USER AJEY;

-- --------------------------------------------
-- --------- Create Warehouse -----------------
-- --------------------------------------------

CREATE WAREHOUSE doc_ai_qs_wh;

-- Give the doc_ai_qs_role role access to the warehouse
GRANT USAGE, OPERATE, MODIFY ON WAREHOUSE doc_ai_qs_wh TO ROLE doc_ai_qs_role;

-- --------------------------------------------
-- --------- Create Database and Schema -------
-- --------------------------------------------

CREATE DATABASE doc_ai_qs_db;
GRANT CREATE SCHEMA, MODIFY, USAGE ON DATABASE doc_ai_qs_db TO ROLE doc_ai_qs_role;

-- Switch to the newly created role
USE ROLE doc_ai_qs_role;

-- Create a schema for the Document AI model, stage, etc.
CREATE SCHEMA doc_ai_qs_db.doc_ai_schema;

-- Grant permissions on the schema
GRANT USAGE ON SCHEMA doc_ai_qs_db.doc_ai_schema TO ROLE doc_ai_qs_role;
GRANT CREATE snowflake.ml.document_intelligence ON SCHEMA doc_ai_qs_db.doc_ai_schema TO ROLE doc_ai_qs_role;

-- --------------------------------------------
-- --------- Create Stage ---------------------
-- --------------------------------------------

-- Create a stage for storing documents
CREATE STAGE doc_ai_qs_db.doc_ai_schema.doc_ai_stage
  DIRECTORY = (ENABLE = true)
  ENCRYPTION = (TYPE = 'snowflake_sse');
