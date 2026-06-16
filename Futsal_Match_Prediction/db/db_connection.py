from sshtunnel import SSHTunnelForwarder
from sqlalchemy import create_engine


import os

class DB:
    def __init__(self):
        self.ssh_host = "13.239.79.237"
        self.ssh_port = 22
        self.ssh_user = "ubuntu"
        
        # Dynamically resolve path to the PEM key so it works locally and inside Docker
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.ssh_key = os.path.join(base_dir, "db_pass", "FutsalozApp_ProdNew.pem")

        self.mysql_host = "futsalozdb.ce8imxcnx4zv.ap-southeast-2.rds.amazonaws.com"
        self.mysql_port = 3306
        self.mysql_user = "admin"
        self.mysql_password = "Dk2poYagiSAUu04wEBtH"
        self.mysql_database = "futsaloz"

        self.tunnel = None
        self.engine = None

    def db_test_connection(self):
        try:
            self.tunnel = SSHTunnelForwarder(
                (self.ssh_host, self.ssh_port),
                ssh_username=self.ssh_user,
                ssh_pkey=self.ssh_key,
                remote_bind_address=(self.mysql_host, self.mysql_port)
            )

            self.tunnel.start()
            print("SSH Tunnel Connected")
            return True

        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def db_create_engine(self):
        if self.tunnel is None or not self.tunnel.is_active:
            print("SSH tunnel is not active. Call db_test_connection() first.")
            return None
            
        self.engine = create_engine(
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@127.0.0.1:{self.tunnel.local_bind_port}/{self.mysql_database}"
        )

        return self.engine

    def db_close_connection(self):
        if self.tunnel:
            self.tunnel.stop()
            print("SSH tunnel closed.")


# Usage
# db = DB()

# db.db_test_connection()
# engine = db.db_create_engine()

# # Test query
# # import pandas as pd
# # df = pd.read_sql("SELECT 1", engine)
# # print(df)

# db.close_connection()