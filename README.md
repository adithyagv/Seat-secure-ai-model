Method : POST

URL : http://localhost:5000/api/v1/predict

Sample Input : 

{
    "title": "Tech Expo 2025",
    "type": "Workshop",
    "price": 1200,
    "city": "Chennai",
    "day_of_week": 3
}

Sample Output : 

{
  "DBSCAN_Cluster": -1,
  "GMM_Cluster": 1,
  "KMeans_Cluster": 1,
  "recommended_events": [
    {
      "city": "Chennai",
      "title": "Cricket League Finals",
      "type": "Sports"
    },
    {
      "city": "Chennai",
      "title": "Startup Pitch Day",
      "type": "Workshop"
    }
  ]
}