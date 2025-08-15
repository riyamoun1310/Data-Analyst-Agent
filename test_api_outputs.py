import requests

# Test weather CSV
with open('test_sample_weather.csv', 'rb') as f:
    resp = requests.post('http://localhost:8000/api/', files={'file': f})
    print('Weather CSV Output:')
    print(resp.json())

# Test sales CSV
with open('test_sample_sales.csv', 'rb') as f:
    resp = requests.post('http://localhost:8000/api/', files={'file': f})
    print('\nSales CSV Output:')
    print(resp.json())
