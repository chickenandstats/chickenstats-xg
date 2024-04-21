#!/bin/bash
set -e
export PGPASSWORD=$POSTGRES_PASSWORD;
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
  CREATE USER "$APP_DB_USER" WITH PASSWORD '$APP_DB_PASSWORD';
  CREATE DATABASE "$APP_DB" WITH OWNER "$APP_DB_USER";
  GRANT ALL PRIVILEGES ON DATABASE mlflow TO "$APP_DB_USER";
  CREATE DATABASE optuna WITH OWNER "$APP_DB_USER";
  GRANT ALL PRIVILEGES ON DATABASE optuna TO "$APP_DB_USER";

EOSQL

psql --username "$APP_DB_USER" mlflow < /backups/mlflow.sql

#psql --username "$APP_DB_USER" optuna < /backups/optuna.sql