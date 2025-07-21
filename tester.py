import pyodbc
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 18 for SQL Server};"
    "SERVER=192.168.0.133,1433;"
    "DATABASE=Reportes;"
    "UID=procesosia;"
    "PWD=Clavepersonal01*;"
    "Encrypt=no;"
    "TrustServerCertificate=yes;"
)
print(conn.cursor().execute("SELECT 1").fetchval())
