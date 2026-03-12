# src/search.py
import re
import numpy as np
from src.config import load_settings, Settings
import torch
from torch import nn

class DeepRecipeEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        super(DeepRecipeEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim) # 128D output
        )

    def forward(self, x):
        # Assuming Global Average Pooling on the sequence dimension
        embedded = self.embedding(x).mean(dim=1) 
        return self.network(embedded)

class QueryEncoder:
    def __init__(self, s):
        self.s = s
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load vocab used for the Mar 12, 14:47 run
        with open("models/vocab_deep_all_features.pkl", "rb") as f:
            self.vocab = pickle.load(f)

        self.model = DeepRecipeEncoder(
            vocab_size=len(self.vocab),
            embedding_dim=128, 
            output_dim=128
        ).to(self.device)

        # Loading the specific Huber-loss weights
        weights_path = "models/deep_huber_all_features_20260312.pt"
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

def parse_user_intent(raw_query):
    """
    Intercepts the raw text and extracts numeric constraints.
    """
    intent = {
        "raw_query": raw_query,
        "clean_text": raw_query.strip(),
        "max_minutes": None,
        "target_minutes": None,
        "proteins": [],
        "dietary_tags": [],
        "courses": [],
        "cuisines": [],
        "methods": [],
        "occasions": []
    }
    
    # 1. Check for hard limits: "under 30 mins", "less than 45 min"
    under_match = re.search(r'(?:under|less than|max)\s*(\d+)\s*(?:min|minute)s?', raw_query, re.IGNORECASE)
    if under_match:
        intent["max_minutes"] = int(under_match.group(1))
        # Remove the constraint from the text so BM25 doesn't search for "under"
        intent["clean_text"] = (raw_query[:under_match.start()] + raw_query[under_match.end():]).strip()

    # 2. Check for proximity: "around 30 mins", "about 45 min"
    around_match = re.search(r'(?:around|about|approx)\s*(\d+)\s*(?:min|minute)s?', raw_query, re.IGNORECASE)
    if around_match:
        intent["target_minutes"] = int(around_match.group(1))
        intent["clean_text"] = (raw_query[:around_match.start()] + raw_query[around_match.end():]).strip()
    
    protein_keywords = ["chicken", "beef", "pork", "tofu", "fish", "turkey", "lamb", "duck", "seafood", "shrimp", "salmon", "tuna", "eggs"]
    for protein in protein_keywords:
        if re.search(r'\b' + re.escape(protein) + r'\b', raw_query, re.IGNORECASE):
            intent["proteins"].append(protein)
    
    tag_mappings = {
        "dietary_tags": {
            r'\bvegan\b': 'vegan',
            r'\bvegetarian\b': 'vegetarian',
            r'\bmeatless\b': 'vegetarian',
            r'\bplant[-\s]?based\b': 'vegan',
            r'\bgluten[-\s]?free\b': 'gluten-free',
            r'\bdairy[-\s]?free\b': 'dairy-free',
            r'\blactose[-\s]?free\b': 'dairy-free',
            r'\begg[-\s]?free\b': 'egg-free',
            r'\bnut[-\s]?free\b': 'nut-free',
            r'\blow[-\s]?carb\b': 'low-carb',
            r'\bvery[-\s]?low[-\s]?carb\b': 'very-low-carbs',
            r'\blow[-\s]?calorie\b': 'low-calorie',
            r'\blow[-\s]?cal\b': 'low-calorie',
            r'\blow[-\s]?fat\b': 'low-fat',
            r'\blow[-\s]?saturated[-\s]?fat\b': 'low-saturated-fat',
            r'\blow[-\s]?sodium\b': 'low-sodium',
            r'\blow[-\s]?cholesterol\b': 'low-cholesterol',
            r'\blow[-\s]?protein\b': 'low-protein',
            r'\bhigh[-\s]?protein\b': 'high-protein',
            r'\bhigh[-\s]?calcium\b': 'high-calcium',
            r'\bhigh[-\s]?fiber\b': 'high-fiber',
            r'\bdiabetic\b': 'diabetic',
            r'\bketo\b': 'low-carb',
            r'\bhealthy\b': 'healthy',
            r'\bkosher\b': 'kosher'
        },
        "courses": {
            r'\bbreakfast\b': 'breakfast',
            r'\bbrunch\b': 'brunch',
            r'\blunch\b': 'lunch',
            r'\bdinner\b': 'main-dish',
            r'\bappetizer[s]?\b': 'appetizers',
            r'\bsnack[s]?\b': 'snacks',
            r'\bdessert[s]?\b': 'desserts',
            r'\bside[-\s]?dish(?:es)?\b': 'side-dishes',
            r'\bsoup[s]?\b': 'soups-stews',
            r'\bsalad[s]?\b': 'salads',
            r'\bbeverage[s]?\b': 'beverages',
            r'\bdrink[s]?\b': 'beverages'
        },
        "cuisines": {
            r'\bmexican\b': 'mexican',
            r'\bitalian\b': 'italian',
            r'\basian\b': 'asian',
            r'\bchinese\b': 'chinese',
            r'\bindian\b': 'indian',
            r'\bthai\b': 'thai',
            r'\bgreek\b': 'greek',
            r'\bfrench\b': 'french',
            r'\bspanish\b': 'spanish',
            r'\bmediterranean\b': 'european',
            r'\bcajun\b': 'cajun',
            r'\bsouthern\b': 'southern-united-states'
        },
        "methods": {
            r'\bcrock[-\s]?pot\b': 'crock-pot-slow-cooker',
            r'\bslow[-\s]?cooker\b': 'crock-pot-slow-cooker',
            r'\binstant[-\s]?pot\b': 'pressure-cooker',
            r'\bmicrowave\b': 'microwave',
            r'\bgrill(?:ing|ed)?\b': 'grilling',
            r'\bbbq\b': 'barbecue',
            r'\bno[-\s]?cook\b': 'no-cook',
            r'\b(?:one|1)[-\s]?pot\b': 'one-dish-meal',
            r'\b(?:one|1)[-\s]?pan\b': 'one-dish-meal'
        },
        "occasions": {
            r'\bthanksgiving\b': 'thanksgiving',
            r'\bchristmas\b': 'christmas',
            r'\bsuper[-\s]?bowl\b': 'superbowl',
            r'\bpotluck\b': 'potluck',
            r'\bcamping\b': 'camping',
            r'\bkid[-\s]?friendly\b': 'kid-friendly',
            r'\btoddler[-\s]?friendly\b': 'toddler-friendly'
        }
    }
    
    for category, mapping in tag_mappings.items():
        for pattern, tag in mapping.items():
            if re.search(pattern, raw_query, re.IGNORECASE):
                if tag not in intent[category]:
                    intent[category].append(tag)
                intent["clean_text"] = re.sub(pattern, '', intent["clean_text"], flags=re.IGNORECASE).strip()
    
    intent["clean_text"] = re.sub(r'\s+', ' ', intent["clean_text"]).strip()
    return intent


def execute_dynamic_search(s, encoder, raw_query, top_k=5):
    """
    Dynamically builds and executes the Elasticsearch JSON payload.
    """
    intent = parse_user_intent(raw_query)
    clean_text = intent["clean_text"]
    
    # Failsafe: If the user just typed "under 30 mins", we need SOME text to search
    if not clean_text:
        clean_text = "recipe"
    
    query_vector = encoder.encode(raw_query)

    # --- Build the Base Lexical Query (BM25) ---
    base_query = {
        "bool": {
            "must": [
                {
                    "multi_match": {
                        "query": clean_text,
                        "fields": ["name^2", "ingredients_clean", "description_clean"],
                        "boost": 0.5
                    }
                }
            ],
            "filter": []
        }
    }

    # --- Apply Hard Filters ---
    # Time limits are usually dealbreakers
    if intent["max_minutes"] is not None:
        base_query["bool"]["filter"].append(
            {"range": {"minutes": {"lte": intent["max_minutes"]}}}
        )
    
    # Dietary requirements are absolute dealbreakers
    for tag in intent["dietary_tags"]:
        base_query["bool"]["filter"].append(
            {"term": {"tags_clean": tag}}
        )

    # --- Apply Soft Filters ---
    # Proteins
    # Dietary tags should handle some protein filtering, but if the user explicitly mentioned "chicken" or "tofu", we should boost those results
    for protein in intent["proteins"]:
        base_query["bool"].setdefault("should", []).append(
            {"match": {"ingredients_clean": {"query": protein, "boost": 2.0}}}
        )

    # Courses, Cuisines, Methods, Occasions
    soft_tags = intent["courses"] + intent["cuisines"] + intent["methods"] + intent["occasions"]
    for tag in soft_tags:
        base_query["bool"].setdefault("should", []).append(
            {"term": {"tags_clean": {"value": tag, "boost": 2.0}}}
        )

    # --- Apply Proximity Decay ---
    final_query = base_query
    
    if intent["target_minutes"] is not None:
        final_query = {
            "function_score": {
                "query": base_query,
                "functions": [
                    {
                        "gauss": {
                            "minutes": {
                                "origin": str(intent["target_minutes"]),
                                "offset": "5",
                                "scale": "15"
                            }
                        }
                    }
                ],
                "boost_mode": "multiply"
            }
        }

    # --- Apply Semantic Vector Search (kNN) ---
    search_payload = {
        "size": top_k,
        "knn": {
            "field": "recipe_embedding",
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": 50,
            "boost": 0.5,
            "filter": base_query["bool"]["filter"]
        },
        "query": final_query
    }

    response = s.es_client.search(
        index=s.index_name,
        **search_payload
    )
    
    return response['hits']['hits'], intent


if __name__ == "__main__":
    s = load_settings()
    encoder = QueryEncoder(s)
    
    # Stress-testing the new Multi-Category Parser
    test_queries = [
        "easy healthy mexican chicken crockpot dinner under 45 mins",
        "vegan thanksgiving sides",
        "low carb italian beef skillet"
    ]
    
    for raw_query in test_queries:
        results, intent = execute_dynamic_search(s, encoder, raw_query)
        
        print("\n" + "="*70)
        print(f"RAW QUERY: '{raw_query}'")
        print(f"BM25 TEXT: '{intent['clean_text']}'")
        print(f"FILTERS  : Time < {intent['max_minutes']} | Proteins={intent['proteins']} | Diets={intent['dietary_tags']}")
        print(f"           Courses={intent['courses']} | Cuisines={intent['cuisines']} | Methods={intent['methods']} | Occasions={intent['occasions']}")
        print("-" * 70)
        
        for idx, hit in enumerate(results):
            score = hit['_score']
            name = hit['_source']['name']
            mins = hit['_source']['minutes']
            print(f"{idx + 1}. [Score: {score:7.3f}] {name.title()} ({mins} mins)")