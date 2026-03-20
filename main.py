from src.config import load_settings
from src.engine import SearchMode
from enum import Enum

# Main entry point for running different components of the project via a CLI menu.
# Provides a simple demo interface for the search engine and intent parsing,
# as well as a way to trigger the data ingestion and evaluation pipelines
# without needing to run separate scripts.

class MainMenuOption(Enum):
    INGESTION = "1"
    QUERY_DEMO = "2"
    INTENT_PARSING_DEMO = "3"
    EVALUATE = "4"
    EXIT = "5"


def main_menu():
    s = load_settings()

    while True:
        print("\nSelect an option:")
        print("1. Run data ingestion pipeline")
        print("2. Run search engine demo with a custom query")
        print("3. Demo intent parsing and weight profiling")
        print("4. Run full evaluation of all search modes")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ").strip()

        try:
            if choice == MainMenuOption.INGESTION.value:
                from src.indexer import run_ingestion
                run_ingestion(s)

            elif choice == MainMenuOption.QUERY_DEMO.value:
                from src.engine import SearchEngine
                engine = SearchEngine(s)

                query = input("Enter your search query: ").strip()
                if not query:
                    print("No query entered.")
                    continue

                print("\nSelect a search mode:")
                for i, mode in enumerate(SearchMode, start=1):
                    print(f"  {i}. {mode.name}")
                mode_choice = input("Enter your choice (1-{}): ".format(len(SearchMode))).strip()

                try:
                    mode_index = int(mode_choice) - 1
                    search_mode = list(SearchMode)[mode_index]
                except (ValueError, IndexError):
                    print("Invalid choice. Using default hybrid mode.")
                    search_mode = SearchMode.HYBRID

                results = engine.run(query=query, mode=search_mode, top_k=5)
                w = results.weights

                print(f"\n{'='*65}")
                print(f"QUERY : {query}")
                print(f"MODE  : {search_mode.value}  |  "
                      f"lex={w['lex']}  align={w['alignment']}  "
                      f"sem={w['semantic']}  quality={w['quality']}")
                print(f"TIER  : {results.tier}")
                print(f"{'='*65}")

                for i, r in enumerate(results.results, start=1):
                    name = r.source.get("name", "").title()
                    mins = r.source.get("minutes")
                    print(
                        f"  {i}. [{r.final_score:.3f}] {name} ({mins}m)"
                        + (
                            f"  align={r.alignment_score:.2f}"
                            f"  sim={r.semantic_sim:.3f}"
                            f"  quality={r.quality_score:.2f}"
                            if search_mode != SearchMode.LEXICAL else ""
                        )
                    )

            elif choice == MainMenuOption.INTENT_PARSING_DEMO.value:
                from src.search import parse_user_intent
                from src.query_encoding import QueryFeatureProjector
                from src.reranker import SemanticReranker

                query = input("Enter a query to parse: ").strip()
                if not query:
                    print("No query entered.")
                    continue

                intent = parse_user_intent(query)
                projector = QueryFeatureProjector(s)
                reranker = SemanticReranker(s)
                projected = projector.project(query, intent)
                weights = reranker.get_weight_profile(projected)

                print(f"\n{'='*55}")
                print(f"QUERY  : {query}")
                print(f"{'='*55}")
                print(f"\nIntent Tier  : {weights['tier'].upper()}")
                print(f"Weight Profile:")
                print(f"  lex={weights['lex']}  "
                      f"align={weights['alignment']}  "
                      f"sem={weights['semantic']}  "
                      f"quality={weights['quality']}")

                print(f"\nParsed Structured Intent:")
                intent_fields = [
                    ("proteins",      "Proteins"),
                    ("dietary_tags",  "Dietary"),
                    ("cuisines",      "Cuisines"),
                    ("courses",       "Courses"),
                    ("methods",       "Methods"),
                    ("occasions",     "Occasions"),
                    ("taste",         "Taste"),
                    ("dish_type",     "Dish Type"),
                    ("max_minutes",   "Max Time"),
                    ("target_minutes","Target Time"),
                ]
                found_any = False
                for key, label in intent_fields:
                    val = intent.get(key)
                    if val:
                        print(f"  {label:<16} {val}")
                        found_any = True
                if not found_any:
                    print("  No structured intent detected — low intent query.")
                    print("  The semantic embedding signal will carry the load.")

                print(f"\nActive Tag Features : {projected.active_tag_features or 'none'}")
                print(f"Active Meta Features: {projected.active_meta_features[:5]}"
                      + (" ..." if len(projected.active_meta_features) > 5 else ""))

            elif choice == MainMenuOption.EVALUATE.value:
                from src.evaluate import evaluate_engine, print_summary
                print("\nRunning five-mode ablation evaluation (~60 seconds)...\n")
                summary = evaluate_engine(top_k=5)
                print_summary(summary)

            elif choice == MainMenuOption.EXIT.value:
                print("Exiting...")
                break

            else:
                print("Invalid choice. Please enter a number between 1 and 5.")

        except KeyboardInterrupt:
            print("\nReturning to main menu...")
            continue
        except Exception as e:
            print(f"\nError: {e}")
            print("Returning to main menu...")
            continue


if __name__ == "__main__":
    main_menu()