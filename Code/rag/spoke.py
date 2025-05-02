import requests
from config.config import config
import ast
import pandas as pd
import logging
import json

spoke_api_logger = logging.getLogger("spoke_api_logger")
spoke_api_logger.setLevel(logging.INFO)

if not spoke_api_logger.handlers:
    file_handler = logging.FileHandler("logs/spoke_api.log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    spoke_api_logger.addHandler(file_handler)


def get_spoke_api_response(base_url, end_point, params=None):
    url = base_url + end_point
    try:
        response = requests.get(url=url, params=params) if params else requests.get(url=url)
        response.raise_for_status()
        log_api_call(spoke_api_logger, url, params, response=response)
        return response
    except requests.exceptions.RequestException as e:
        log_api_call(spoke_api_logger, url, params, error=str(e))
        raise


def get_data_types_from_spoke_api():
    end_point = "/api/v1/types"
    response = get_spoke_api_response(config.BASE_URL, end_point)
    data_types = response.json()
    node_types = list(data_types["nodes"].keys())
    edge_types = list(data_types["edges"].keys())
    return node_types, edge_types


def get_context_from_spoke_api(node):
    node_types, edge_types = get_data_types_from_spoke_api()
    node_types_to_remove = ["DatabaseTimestamp", "Version"]
    filtered_node_types = [node_type for node_type in node_types if node_type not in node_types_to_remove]

    api_params = {
        'node_filters' : filtered_node_types,
        'edge_filters': edge_types,
        'cutoff_Compound_max_phase': config.CUTOFF_COMPOUND_MAX_PHASE,
        'cutoff_Protein_source': config.CUTOFF_PROTEIN_SOURCE,
        'cutoff_DaG_diseases_sources': config.CUTOFF_DAG_DISEASE_SOURCES,
        'cutoff_DaG_textmining': config.CUTOFF_DAG_TERMINATING,
        'cutoff_CtD_phase': config.CUTOFF_CTD_PHASE,
        'cutoff_PiP_confidence': config.CUTOFF_PIP_CONFIDENCE,
        'cutoff_ACTeG_level': config.CUTOFF_ACTEG_LEVEL,
        'cutoff_DpL_average_prevalence': config.CUTOFF_DPL_AVERAGE_PREVALENCE,
        'depth' : config.DEPTH
    }

    node_type = "Disease"
    attribute = "name"
    neighbour_end_point = f"/api/v1/neighborhood/{node_type}/{attribute}/{node}"
    response = get_spoke_api_response(config.BASE_URL, neighbour_end_point, params=api_params)
    node_context = response.json()

    neighbour_nodes = []
    neighbour_edges = []

    for item in node_context:
        if "_" not in item["data"]["neo4j_type"]:
            try:
                if item["data"]["neo4j_type"] == "Protein":
                    neighbour_nodes.append((item["data"]["neo4j_type"], item["data"]["id"], item["data"]["properties"]["description"]))
                else:
                    neighbour_nodes.append((item["data"]["neo4j_type"], item["data"]["id"], item["data"]["properties"]["name"]))

            except:
                neighbour_nodes.append((item["data"]["neo4j_type"], item["data"]["id"], item["data"]["properties"]["identifier"]))
        
        elif "_" in item["data"]["neo4j_type"]:
            try:
                provenance = ", ".join(item["data"]["properties"]["sources"])

            except:
                try:
                    provenance = item["data"]["properties"]["source"]
                    if isinstance(provenance, list):
                        provenance = ", ".join(provenance)                    
                except:
                    try:                    
                        preprint_list = ast.literal_eval(item["data"]["properties"]["preprint_list"])
                        if len(preprint_list) > 0:                                                    
                            provenance = ", ".join(preprint_list)
                        else:
                            pmid_list = ast.literal_eval(item["data"]["properties"]["pmid_list"])
                            pmid_list = map(lambda x:"pubmedId:"+x, pmid_list)
                            if len(pmid_list) > 0:
                                provenance = ", ".join(pmid_list)
                            else:
                                provenance = "Based on data from Institute For Systems Biology (ISB)"
                    except:                                
                        provenance = "SPOKE-KG"     
            try:
                evidence = item["data"]["properties"]
            except:
                evidence = None
            neighbour_edges.append((item["data"]["source"], item["data"]["neo4j_type"], item["data"]["target"], provenance, evidence))


    neighbour_nodes_df = pd.DataFrame(neighbour_nodes, columns=["node_type", "node_id", "node_name"])
    neighbour_edges_df = pd.DataFrame(neighbour_edges, columns=["source", "edge_type", "target", "provenance", "evidence"])
    context, merged_df = generate_context_and_merged_data(neighbour_nodes_df, neighbour_edges_df, node, node_context)
    return context, merged_df


def generate_context_and_merged_data(nodes_df, edges_df, node_value, node_context):
    merged_with_source = pd.merge(
        edges_df, 
        nodes_df, 
        left_on="source", 
        right_on="node_id"
    ).drop("node_id", axis=1)

    merged_with_source["source_name"] = (
        merged_with_source["node_type"] + " " + merged_with_source["node_name"]
    )
    merged_with_source.drop(["source", "node_type", "node_name"], axis=1, inplace=True)
    merged_with_source.rename(columns={"source_name": "source"}, inplace=True)

    merged_with_target = pd.merge(
        merged_with_source, 
        nodes_df, 
        left_on="target", 
        right_on="node_id"
    ).drop("node_id", axis=1)

    merged_with_target["target_name"] = (
        merged_with_target["node_type"] + " " + merged_with_target["node_name"]
    )
    merged_with_target.drop(["target", "node_type", "node_name"], axis=1, inplace=True)
    merged_with_target.rename(columns={"target_name": "target"}, inplace=True)

    final_merged_df = merged_with_target[
        ["source", "edge_type", "target", "provenance", "evidence"]
    ].copy()

    final_merged_df.loc[:, "predicate"] = final_merged_df["edge_type"].apply(
        lambda x: x.split("_")[0]
    )

    final_merged_df.loc[:, "context"] = (
        final_merged_df["source"] + " " +
        final_merged_df["predicate"].str.lower() + " " +
        final_merged_df["target"] + "."
    )

    combined_context = final_merged_df["context"].str.cat(sep=" ")

    combined_context += (
        f" {node_value} has a {node_context[0]['data']['properties']['source']} "
        f"identifier of {node_context[0]['data']['properties']['identifier']}."
    )

    return combined_context, final_merged_df


def log_api_call(logger, url, params=None, response=None, error=None):
    log_data = {
        "url": url,
        "params": params if params else {},
    }

    if error:
        log_data["status"] = "failure"
        log_data["error"] = error
        logger.error(json.dumps(log_data, indent=4))
    else:
        log_data["status"] = "success"
        log_data["status_code"] = response.status_code
        log_data["content_length"] = len(response.content)
        logger.info(json.dumps(log_data, indent=4))