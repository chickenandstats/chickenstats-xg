#!/bin/bash
set -e
export PGPASSWORD=$POSTGRES_PASSWORD;
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
  CREATE USER "$APP_DB_USER" WITH PASSWORD '$APP_DB_PASSWORD';
  CREATE DATABASE evolving_hockey WITH OWNER "$APP_DB_USER";
  GRANT ALL PRIVILEGES ON DATABASE evolving_hockey TO "$APP_DB_USER";
  CREATE DATABASE chicken_nhl WITH OWNER "$APP_DB_USER";
  GRANT ALL PRIVILEGES ON DATABASE chicken_nhl TO "$APP_DB_USER";
  CREATE DATABASE users WITH OWNER "$APP_DB_USER";
  GRANT ALL PRIVILEGES ON DATABASE users TO "$APP_DB_USER";

EOSQL

#psql --username "$APP_DB_USER" mlflow < /backups/mlflow.sql

#psql --username "$APP_DB_USER" optuna < /backups/optuna.sql