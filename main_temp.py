from YDD_Client import YDDClient

client = YDDClient()
result = client.query_yundan_detail(['DT0010018393US'])
print(result)