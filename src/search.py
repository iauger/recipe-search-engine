# src/search.py
import re
from enum import Enum
from typing import Any, Dict, List, Tuple

from src.config import load_settings


class SearchField(str, Enum):
    NAME = "name^2"
    INGREDIENTS = "ingredients_clean"
    DESCRIPTION = "description_clean"
    TAGS = "tags_clean"
    MINUTES = "minutes"


class BoostValue(float, Enum):
    MULTI_MATCH = 0.5
    PROTEIN_MATCH = 2.0
    TAG_MATCH = 1.5
    TASTE_MATCH = 0.5


class RetrievalConfig(int, Enum):
    DEFAULT_TOP_K = 10
    TARGET_MINUTES_OFFSET = 5
    TARGET_MINUTES_SCALE = 15


class IntentKey(str, Enum):
    RAW_QUERY = "raw_query"
    LEXICAL_QUERY = "lexical_query"
    CLEAN_TEXT = "clean_text"
    MAX_MINUTES = "max_minutes"
    TARGET_MINUTES = "target_minutes"
    PROTEINS = "proteins"
    DIETARY_TAGS = "dietary_tags"
    COURSES = "courses"
    CUISINES = "cuisines"
    METHODS = "methods"
    OCCASIONS = "occasions"
    TASTE = "taste"
    DISH_TYPE = "dish_type"


PROTEIN_KEYWORDS = [
    "chicken",
    "beef",
    "pork",
    "tofu",
    "fish",
    "turkey",
    "lamb",
    "duck",
    "seafood",
    "shrimp",
    "salmon",
    "tuna",
    "eggs",
]

TAG_MAPPINGS: Dict[str, Dict[str, str]] = {
    IntentKey.DIETARY_TAGS.value: {
        r"\bvegan\b": "vegan",
        r"\bvegetarian\b": "vegetarian",
        r"\bmeatless\b": "vegetarian",
        r"\bplant[-\s]?based\b": "vegan",
        r"\bgluten[-\s]?free\b": "gluten-free",
        r"\bdairy[-\s]?free\b": "dairy-free",
        r"\blactose[-\s]?free\b": "dairy-free",
        r"\begg[-\s]?free\b": "egg-free",
        r"\bnut[-\s]?free\b": "nut-free",
        r"\blow[-\s]?carb\b": "low-carb",
        r"\bvery[-\s]?low[-\s]?carb\b": "very-low-carbs",
        r"\blow[-\s]?calorie\b": "low-calorie",
        r"\blow[-\s]?cal\b": "low-calorie",
        r"\blow[-\s]?fat\b": "low-fat",
        r"\blow[-\s]?saturated[-\s]?fat\b": "low-saturated-fat",
        r"\blow[-\s]?sodium\b": "low-sodium",
        r"\blow[-\s]?cholesterol\b": "low-cholesterol",
        r"\blow[-\s]?protein\b": "low-protein",
        r"\bhigh[-\s]?protein\b": "high-protein",
        r"\bhigh[-\s]?calcium\b": "high-calcium",
        r"\bhigh[-\s]?fiber\b": "high-fiber",
        r"\bdiabetic\b": "diabetic",
        r"\bketo\b": "low-carb",
        r"\bhealthy\b": "healthy",
        r"\bkosher\b": "kosher",
    },
    IntentKey.COURSES.value: {
        r"\bbreakfast\b": "breakfast",
        r"\bbrunch\b": "brunch",
        r"\blunch\b": "lunch",
        r"\bdinner\b": "main-dish",
        r"\bappetizer[s]?\b": "appetizers",
        r"\bsnack[s]?\b": "snacks",
        r"\bdessert[s]?\b": "desserts",
        r"\b(side[-\s]?dish(?:es)?|side(?:s)?|accompaniment(?:s)?)\b": "side-dishes",
        r"\bsoup[s]?\b": "soups-stews",
        r"\bsalad[s]?\b": "salads",
        r"\bbeverage[s]?\b": "beverages",
        r"\bdrink[s]?\b": "beverages",
    },
    IntentKey.CUISINES.value: {
        r"\bmexican\b": "mexican",
        r"\bitalian\b": "italian",
        r"\basian\b": "asian",
        r"\bchinese\b": "chinese",
        r"\bindian\b": "indian",
        r"\bthai\b": "thai",
        r"\bgreek\b": "greek",
        r"\bfrench\b": "french",
        r"\bspanish\b": "spanish",
        r"\bmediterranean\b": "european",
        r"\bcajun\b": "cajun",
        r"\bsouthern\b": "southern-united-states",
    },
    IntentKey.METHODS.value: {
        r"\bcrock[-\s]?pot\b": "crock-pot-slow-cooker",
        r"\bslow[-\s]?cooker\b": "crock-pot-slow-cooker",
        r"\binstant[-\s]?pot\b": "pressure-cooker",
        r"\bmicrowave\b": "microwave",
        r"\bgrill(?:ing|ed)?\b": "grilling",
        r"\bbbq\b": "barbecue",
        r"\bno[-\s]?cook\b": "no-cook",
        r"\b(?:one|1)[-\s]?pot\b": "one-dish-meal",
        r"\b(?:one|1)[-\s]?pan\b": "one-dish-meal",
    },
    IntentKey.OCCASIONS.value: {
        r"\bthanksgiving\b": "thanksgiving",
        r"\bchristmas\b": "christmas",
        r"\bsuper[-\s]?bowl\b": "superbowl",
        r"\bpotluck\b": "potluck",
        r"\bcamping\b": "camping",
        r"\bkid[-\s]?friendly\b": "kid-friendly",
        r"\btoddler[-\s]?friendly\b": "toddler-friendly",
    },
    IntentKey.TASTE.value: {
    r"\bsweet\b": "sweet",
    r"\bspicy\b": "spicy",
    r"\bsavory\b": "savory",
    r"\bsalty\b": "salty",
    r"\bsour\b": "sour",
    r"\btangy\b": "tangy",
    r"\bsmoky\b": "smoky",
    r"\bcreamy\b": "creamy",
    r"\bgarlicky\b": "garlicky",
    r"\bcheesy\b": "cheesy",
    },
    IntentKey.DISH_TYPE.value: {
    r"\btaco[s]?\b": "tacos",
    r"\bburrito[s]?\b": "burritos",
    r"\bwrap[s]?\b": "wraps",
    r"\bsandwich(?:es)?\b": "sandwiches",
    r"\bburger[s]?\b": "burgers",
    r"\bpizza\b": "pizza",
    r"\bpasta\b": "pasta",
    r"\bskillet\b": "skillet",
    r"\bsoup[s]?\b": "soups",
    r"\bstew[s]?\b": "stews",
    r"\bsalad[s]?\b": "salads",
}
}


def initialize_intent(raw_query: str) -> Dict[str, Any]:
    return {
        IntentKey.RAW_QUERY.value: raw_query,
        IntentKey.LEXICAL_QUERY.value: raw_query.strip(),
        IntentKey.CLEAN_TEXT.value: raw_query.strip(),
        IntentKey.MAX_MINUTES.value: None,
        IntentKey.TARGET_MINUTES.value: None,
        IntentKey.PROTEINS.value: [],
        IntentKey.DIETARY_TAGS.value: [],
        IntentKey.COURSES.value: [],
        IntentKey.CUISINES.value: [],
        IntentKey.METHODS.value: [],
        IntentKey.OCCASIONS.value: [],
        IntentKey.TASTE.value: [],
        IntentKey.DISH_TYPE.value: [],
    }


def extract_time_constraints(intent: Dict[str, Any], raw_query: str) -> None:
    under_match = re.search(
        r"(?:under|less than|max)\s*(\d+)\s*(?:min|minute)s?",
        raw_query,
        re.IGNORECASE,
    )
    if under_match:
        intent[IntentKey.MAX_MINUTES.value] = int(under_match.group(1))
        intent[IntentKey.CLEAN_TEXT.value] = (
            raw_query[: under_match.start()] + raw_query[under_match.end() :]
        ).strip()

    around_match = re.search(
        r"(?:around|about|approx)\s*(\d+)\s*(?:min|minute)s?",
        raw_query,
        re.IGNORECASE,
    )
    if around_match:
        intent[IntentKey.TARGET_MINUTES.value] = int(around_match.group(1))
        intent[IntentKey.CLEAN_TEXT.value] = (
            intent[IntentKey.CLEAN_TEXT.value][: around_match.start()]
            + intent[IntentKey.CLEAN_TEXT.value][around_match.end() :]
        ).strip()


def extract_proteins(intent: Dict[str, Any], raw_query: str) -> None:
    for protein in PROTEIN_KEYWORDS:
        if re.search(rf"\b{re.escape(protein)}\b", raw_query, re.IGNORECASE):
            intent[IntentKey.PROTEINS.value].append(protein)


def extract_tag_intent(intent: Dict[str, Any], raw_query: str) -> None:
    clean_text = intent[IntentKey.CLEAN_TEXT.value]

    for category, mapping in TAG_MAPPINGS.items():
        for pattern, tag in mapping.items():
            if re.search(pattern, raw_query, re.IGNORECASE):
                if tag not in intent[category]:
                    intent[category].append(tag)
                clean_text = re.sub(pattern, "", clean_text, flags=re.IGNORECASE).strip()

    intent[IntentKey.CLEAN_TEXT.value] = re.sub(r"\s+", " ", clean_text).strip()


def parse_user_intent(raw_query: str) -> Dict[str, Any]:
    """
    Parse user query into:
    - cleaned lexical query text
    - hard constraints
    - soft preferences
    """
    intent = initialize_intent(raw_query)
    extract_time_constraints(intent, raw_query)
    extract_proteins(intent, raw_query)
    extract_tag_intent(intent, raw_query)
    return intent


def build_base_bool_query(intent: Dict[str, Any]) -> Dict[str, Any]:
    lexical_query = intent[IntentKey.LEXICAL_QUERY.value] or "recipe"

    return {
        "bool": {
            "must": [
                {
                    "multi_match": {
                        "query": lexical_query,
                        "fields": [
                            SearchField.NAME.value,
                            SearchField.INGREDIENTS.value,
                            SearchField.DESCRIPTION.value,
                        ],
                        "boost": float(BoostValue.MULTI_MATCH.value),
                    }
                }
            ],
            "filter": [],
            "should": [],
        }
    }


def apply_hard_filters(base_query: Dict[str, Any], intent: Dict[str, Any]) -> None:
    max_minutes = intent[IntentKey.MAX_MINUTES.value]
    if max_minutes is not None:
        base_query["bool"]["filter"].append(
            {
                "range": {
                    SearchField.MINUTES.value: {
                        "lte": max_minutes,
                    }
                }
            }
        )

    for tag in intent[IntentKey.DIETARY_TAGS.value]:
        base_query["bool"]["filter"].append(
            {"term": {SearchField.TAGS.value: tag}}
        )


def apply_soft_boosts(base_query: Dict[str, Any], intent: Dict[str, Any]) -> None:
    for protein in intent[IntentKey.PROTEINS.value]:
        base_query["bool"]["should"].append(
            {
                "match": {
                    SearchField.INGREDIENTS.value: {
                        "query": protein,
                        "boost": float(BoostValue.PROTEIN_MATCH.value),
                    }
                }
            }
        )

    soft_tags = (
        intent[IntentKey.COURSES.value]
        + intent[IntentKey.CUISINES.value]
        + intent[IntentKey.METHODS.value]
        + intent[IntentKey.OCCASIONS.value]
        + intent[IntentKey.DISH_TYPE.value]
    )

    for tag in soft_tags:
        base_query["bool"]["should"].append(
            {
                "term": {
                    SearchField.TAGS.value: {
                        "value": tag,
                        "boost": float(BoostValue.TAG_MATCH.value),
                    }
                }
            }
        )

    for taste in intent[IntentKey.TASTE.value]:
        base_query["bool"]["should"].append(
            {
                "term": {
                    SearchField.TAGS.value: {
                        "value": taste,
                        "boost": float(BoostValue.TASTE_MATCH.value),
                    }
                }
            }
        )


def apply_time_proximity_scoring(
    base_query: Dict[str, Any],
    intent: Dict[str, Any],
) -> Dict[str, Any]:
    target_minutes = intent[IntentKey.TARGET_MINUTES.value]
    if target_minutes is None:
        return base_query

    return {
        "function_score": {
            "query": base_query,
            "functions": [
                {
                    "gauss": {
                        SearchField.MINUTES.value: {
                            "origin": str(target_minutes),
                            "offset": str(RetrievalConfig.TARGET_MINUTES_OFFSET.value),
                            "scale": str(RetrievalConfig.TARGET_MINUTES_SCALE.value),
                        }
                    }
                }
            ],
            "boost_mode": "multiply",
        }
    }


def build_candidate_query(intent: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build first-stage lexical retrieval query only.
    No dense vectors, no query encoder, no reranking logic.
    """
    base_query = build_base_bool_query(intent)
    apply_hard_filters(base_query, intent)
    apply_soft_boosts(base_query, intent)
    return apply_time_proximity_scoring(base_query, intent)


def retrieve_candidates(
    s: Any,
    raw_query: str,
    top_k: int = RetrievalConfig.DEFAULT_TOP_K.value,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Stage 1 retrieval:
    - parse query intent
    - build lexical/filter query
    - retrieve candidate recipes from Elasticsearch

    Returns:
        candidates: ES hits
        intent: parsed intent dictionary
    """
    intent = parse_user_intent(raw_query)
    final_query = build_candidate_query(intent)

    search_payload = {
        "size": top_k,
        "query": final_query,
    }

    response = s.es_client.search(index=s.index_name, **search_payload)
    return response["hits"]["hits"], intent


if __name__ == "__main__":
    s = load_settings()

    test_queries = [
        "easy healthy mexican chicken crockpot dinner under 45 mins",
        "vegan thanksgiving sides",
        "low carb italian beef skillet",
    ]

    for q in test_queries:
        results, intent = retrieve_candidates(s, q)

        print(f"\nQUERY: {q}")
        print("-" * 60)
        print("Parsed intent:", intent)

        for i, hit in enumerate(results, start=1):
            source = hit["_source"]
            name = source.get("name", "").title()
            minutes = source.get("minutes")
            score = hit.get("_score")
            print(f"{i}. [{score:.2f}] {name} ({minutes}m)")