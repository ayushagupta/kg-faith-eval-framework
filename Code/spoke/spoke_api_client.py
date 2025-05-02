import requests
from config.config import config
import ast
import pandas as pd
import logging
import json

class SpokeAPIClient:
    def __init__(self):
        self.base_url = config.BASE_URL
        self.logger = logging.getLogger("spoke_api_logger")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            file_handler = logging.FileHandler("logs/spoke_api.log")
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        self.get_data_types()


    def _log_api_call(self, url, params=None, response=None, error=None):
        log_data = {
            "url": url,
            "params": params if params else {},
        }

        if error:
            log_data["status"] = "failure"
            log_data["error"] = error
            self.logger.error(json.dumps(log_data, indent=4))
        else:
            log_data["status"] = "success"
            log_data["status_code"] = response.status_code
            log_data["content_length"] = len(response.content)
            self.logger.info(json.dumps(log_data, indent=4))

    
    def _get(self, endpoint, params=None):
        url = self.base_url + endpoint
        try:
            response = requests.get(url=url, params=params) if params else requests.get(url=url)
            response.raise_for_status()
            self._log_api_call(url, params, response=response)
            return response
        except requests.exceptions.RequestException as e:
            self._log_api_call(url, params, error=str(e))
            raise


    def get_data_types(self):
        endpoint = "/api/v1/types"
        response = self._get(endpoint)
        data_types = response.json()
        self.node_types = list(data_types["nodes"].keys())
        self.edge_types = list(data_types["edges"].keys())
        return self.node_types, self.edge_types


    def get_context(self, node):
        node_types_to_remove = ["DatabaseTimestamp", "Version"]
        filtered_node_types = [node_type for node_type in self.node_types if node_type not in node_types_to_remove]

        api_params = {
            'node_filters' : filtered_node_types,
            'edge_filters': self.edge_types,
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
        response = self._get(neighbour_end_point, params=api_params)
        node_context = response.json()

        return self._parse_context(node, node_context)
    

    def _parse_context(self, node, node_context):
        neighbour_nodes = []
        neighbour_edges = []

        for item in node_context:
            data = item["data"]
            neo_type = data["neo4j_type"]

            if "_" not in neo_type:
                try:
                    name = data["properties"]["description"] if neo_type == "Protein" else data["properties"]["name"]
                except:
                    name = data["properties"]["identifier"]
                
                neighbour_nodes.append((neo_type, data["id"], name))
            
            else:
                provenance = self._extract_provenance(data["properties"])
                evidence = data.get("properties", None)
                neighbour_edges.append((data["source"], neo_type, data["target"], provenance, evidence))

        nodes_df = pd.DataFrame(neighbour_nodes, columns=["node_type", "node_id", "node_name"])
        edges_df = pd.DataFrame(neighbour_edges, columns=["source", "edge_type", "target", "provenance", "evidence"])

        return self._generate_context_and_merged_data(nodes_df, edges_df, node, node_context)


    def _extract_provenance(self, props):
        try:
            return ", ".join(props["sources"])
        except:
            try:
                source = props["source"]
                return ", ".join(source) if isinstance(source, list) else source
            except:
                try:
                    preprint_list = ast.literal_eval(props.get("preprint_list", "[]"))
                    if preprint_list:
                        return ", ".join(preprint_list)
                    pmid_list = ast.literal_eval(props.get("pmid_list", "[]"))
                    pmid_list = list(map(lambda x: "pubmedId:" + x, pmid_list))
                    return ", ".join(pmid_list) if pmid_list else "Based on data from Institute For Systems Biology (ISB)"
                except:
                    return "SPOKE-KG"
    

    def _generate_context_and_merged_data(self, nodes_df, edges_df, node_value, node_context):
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
