import requests
from config.config import config


def get_spoke_api_response(base_url, end_point, params=None):
    url = base_url + end_point
    if params:
        return requests.get(url=url, params=params)
    else:
        return requests.get(url=url)


def get_data_types_from_spoke_api():
    end_point = "/api/v1/types"
    response = get_spoke_api_response(config.BASE_URL, end_point)
    data_types = response.json()
    node_types = list(data_types["nodes"].keys())
    edge_types = list(data_types["edges"].keys())
    return node_types, edge_types
