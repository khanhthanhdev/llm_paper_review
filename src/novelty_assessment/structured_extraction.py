import os
import json
import glob
import textwrap
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Define models for structured data
class ResultEntry(BaseModel):
    """Represents a single key quantitative result from a paper.

    Attributes:
        metric: The name of the evaluation metric (e.g., "Accuracy", "F1-Score").
        value: The reported value for the metric, stored as a string.
    """

    metric: str = Field(description="Name of the evaluation metric")
    value: str = Field(description="Value of the metric")


class StructuredPaperInformation(BaseModel):
    """A Pydantic model defining the schema for structured data from a paper.

    This model ensures that the information extracted by the LLM is consistent
    and adheres to a predefined format, making it reliable for downstream
    analysis tasks.

    Attributes:
        methods: A list of methods or approaches proposed in the paper.
        problems: A list of problems or tasks the paper addresses.
        datasets: A list of datasets used for evaluation.
        metrics: A list of evaluation metrics used to measure performance.
        results: A list of key quantitative results, captured as `ResultEntry` objects.
        novelty_claims: A list of explicit claims the authors make about the
                        novelty of their work.
    """

    methods: List[str] = Field(
        description="List of methods/approaches proposed in the paper"
    )
    problems: List[str] = Field(description="List of problems the paper addresses")
    datasets: List[str] = Field(description="List of datasets used for evaluation")
    metrics: List[str] = Field(description="List of evaluation metrics used")
    results: List[ResultEntry] = Field(
        description="Key quantitative results with metric name and value"
    )
    novelty_claims: List[str] = Field(
        description="Claims about what is novel in this work"
    )


class StructuredRepresentationGenerator:
    """Extracts structured information from research papers using an LLM.

    This class provides methods to process individual papers or entire submissions,
    extracting key information like methods, problems, datasets, and novelty claims
    based on a paper's title, abstract, and introduction. It supports both
    real-time processing and batch inference preparation.

    Attributes:
        llm: An instance of a LangChain ChatOpenAI model.
        template (str): The prompt template used for the extraction task.
    """

    def __init__(self, model_name: str = "gpt-4.1", temperature: float = 0.0):
        """Initializes the StructuredRepresentationGenerator.

        Args:
            model_name: The identifier for the OpenAI model to be used.
            temperature: The temperature setting for the LLM, controlling output
                         randomness.

        Raises:
            ValueError: If the `OPENAI_API_KEY` environment variable is not set.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.llm = ChatOpenAI(
            model_name=model_name, temperature=temperature, api_key=api_key
        )

        self.template = textwrap.dedent(
            """
        You are tasked with extracting key information from a research paper for building a knowledge graph.

        Paper title: {title}

        Based on the paper content provided below, extract the following information:
        - "methods": [List of methods/approaches proposed in the paper],
        - "problems": [List of problems the paper addresses],
        - "datasets": [List of datasets used for evaluation],
        - "metrics": [List of evaluation metrics used],
        - "results": [List of objects with 'metric' and 'value' fields representing key quantitative results],
        - "novelty_claims": [Claims about what is novel in this work]

        Be precise and specific.

        Paper content:
        {abstract}
        {introduction} 
        """
        )

    def extract_paper_information(
        self, paper_title: str, abstract: str, introduction: str = ""
    ) -> Dict[str, Any]:
        """Extracts structured information from the text of a single research paper.

        This method uses a predefined prompt and the initialized LLM to parse the
        paper's title, abstract, and introduction into the `StructuredPaperInformation`
        schema.

        Args:
            paper_title: The title of the paper.
            abstract: The abstract of the paper.
            introduction: The introduction of the paper (optional but recommended).

        Returns:
            A dictionary containing the raw string response from the LLM and the
            parsed Pydantic object as a dictionary.
        """
        prompt = self.template.format(
            title=paper_title, abstract=abstract, introduction=introduction
        )

        # Run the chain without include_raw to get clean structured output
        result = self.llm.with_structured_output(StructuredPaperInformation).invoke(
            prompt
        )

        # Return clean format
        return {"raw": str(result), "parsed": dict(result)}

    def process_single_submission(self, data_dir: str, submission_id: str) -> Dict[str, Any]:
        """Processes a single submission, including its main paper and related works.

        This method orchestrates the structured extraction for an entire submission
        package. It processes the main paper individually and then uses the
        efficient `batch` method from LangChain to process the top 10 related
        papers concurrently. The results are saved to a JSON file.

        Args:
            data_dir: The base directory where submission data is stored.
            submission_id: The unique identifier for the submission to process.

        Returns:
            A dictionary containing the structured representations for the main
            paper and its selected related papers.
        """
        submission_path = os.path.join(data_dir, submission_id)

        # Load main paper
        with open(f"{submission_path}/{submission_id}.json", "r") as f:
            main_paper = json.load(f)

        # Get introduction if available
        intro = ""
        if os.path.exists(f"{submission_path}/introductions/{submission_id}_intro.txt"):
            with open(f"{submission_path}/introductions/{submission_id}_intro.txt", "r") as f:
                intro = f.read()

        # Process main paper
        main_result = self.extract_paper_information(
            main_paper["title"], main_paper["abstract"], intro
        )

        # Load related papers and prepare prompts
        with open(f"{submission_path}/related_work_data/ranked_papers.json", "r") as f:
            related_papers = json.load(f)

        # Prepare all prompts for batch processing
        prompts = []
        for paper in related_papers[:10]:  # Process top 10
            paper_intro = ""
            intro_path = (
                f"{submission_path}/introductions/{paper['paper_id']}_intro.txt"
            )
            if os.path.exists(intro_path):
                with open(intro_path, "r") as f:
                    paper_intro = f.read()

            prompt = self.template.format(
                title=paper["title"],
                abstract=paper["abstract"],
                introduction=paper_intro,
            )
            prompts.append(prompt)

        # Batch process all related papers
        chain = self.llm.with_structured_output(StructuredPaperInformation)
        related_raw_results = chain.batch(prompts)

        # Process batch results to consistent format
        related_results = []
        for i, result in enumerate(related_raw_results):
            related_results.append(
                {
                    "paper_id": related_papers[i]["paper_id"],
                    "raw": str(result),
                    "parsed": dict(result),
                }
            )

        # Format output with proper serialization
        def serialize_result_entries(results):
            """Convert ResultEntry objects to dicts"""
            if not results:
                return []
            return [
                {"metric": r.metric, "value": r.value} if hasattr(r, "metric") else r
                for r in results
            ]

        # Convert main paper parsed data
        main_parsed = dict(main_result["parsed"])
        if "results" in main_parsed:
            main_parsed["results"] = serialize_result_entries(main_parsed["results"])

        output = {
            "main_paper": {
                "raw": str(
                    main_result["raw"]
                ),  # Convert to string to avoid serialization issues
                "parsed": main_parsed,
            },
            "selected_papers": [],
        }

        # Process related papers
        for r in related_results:
            related_parsed = dict(r["parsed"])
            if "results" in related_parsed:
                related_parsed["results"] = serialize_result_entries(
                    related_parsed["results"]
                )

            output["selected_papers"].append(
                {
                    "paper_id": r["paper_id"],
                    "raw": str(
                        r["raw"]
                    ),  # Convert to string to avoid serialization issues
                    "parsed": related_parsed,
                }
            )

        # Save results
        output_path = f"{submission_path}/structured_representation.json"
        with open(output_path, "w") as f:
            json.dump({"structured_representation": output}, f, indent=4)

        return output

    def prepare_batch_inference_data(
        self,
        data_dir: str,
        model_name: str = "gpt-4.1",
        output_dir: str = "./openai_inputs",
    ) -> str:
        """Prepares a JSONL file for batch processing with the OpenAI API.

        This method iterates through all submission directories, constructs a
        prompt for the main paper and each of its related papers, and formats
        them into a JSONL file suitable for the OpenAI batch API endpoint. This
        is highly efficient for processing a large number of papers.

        Args:
            data_dir: The directory containing all submission subdirectories.
            model_name: The OpenAI model identifier to be included in the batch request.
            output_dir: The directory where the generated batch file will be saved.

        Returns:
            The path to the generated JSONL file.
        """
        batch_entries = []

        # Statistics for reporting
        total_main_papers = 0
        main_papers_with_intro = 0
        total_selected_papers = 0
        selected_papers_with_intro = 0

        submissions = os.listdir(data_dir)
        for submission in submissions:
            # Handle different submission ID formats
            submission_id = (
                submission.split("_")[0] if "_" in submission else submission
            )

            # Check if submission paper exists
            submission_paper_path = f"{data_dir}/{submission}/{submission}.json"
            if not os.path.exists(submission_paper_path):
                print(f"Skipping {submission}: main paper not found")
                continue

            try:
                with open(submission_paper_path, "r") as f:
                    paper_obj = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading {submission_id}: {e}")
                continue

            main_paper_title = paper_obj["title"]
            main_paper_abstract = paper_obj["abstract"]
            main_paper_introduction = ""
            if os.path.exists(
                f"{data_dir}/{submission}/introductions/{submission}_intro.txt"
            ):
                main_paper_introduction = open(
                    f"{data_dir}/{submission}/introductions/{submission}_intro.txt", "r"
                ).read()
                main_papers_with_intro += 1

            total_main_papers += 1

            main_paper_prompt = self.template.format(
                title=main_paper_title,
                abstract=main_paper_abstract,
                introduction=main_paper_introduction,
            )

            from openai.lib._pydantic import to_strict_json_schema

            sch = {
                "name": "StructuredPaperInformation",
                "schema": to_strict_json_schema(StructuredPaperInformation),
            }

            main_paper_entry = {
                "custom_id": f"structured-rep-main-{submission_id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": [{"role": "user", "content": main_paper_prompt}],
                    "temperature": 0.0,
                    "response_format": {"type": "json_schema", "json_schema": sch},
                },
            }

            batch_entries.append(main_paper_entry)

            # Check if related papers exist
            related_papers_path = (
                f"{data_dir}/{submission}/related_work_data/ranked_papers.json"
            )
            if not os.path.exists(related_papers_path):
                print(f"Skipping {submission}: related papers not found")
                continue

            try:
                with open(related_papers_path, "r") as f:
                    selected_papers = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading related papers for {submission_id}: {e}")
                continue

            for paper in selected_papers:
                total_selected_papers += 1
                paper_id = paper["paper_id"]
                paper_title = paper["title"]
                paper_abstract = paper["abstract"]
                if os.path.exists(
                    f"{data_dir}/{submission}/introductions/{paper_id}_intro.txt"
                ):
                    paper_introduction = open(
                        f"{data_dir}/{submission}/introductions/{paper_id}_intro.txt",
                        "r",
                    ).read()
                    selected_papers_with_intro += 1
                else:
                    paper_introduction = ""

                paper_prompt = self.template.format(
                    title=paper_title,
                    abstract=paper_abstract,
                    introduction=paper_introduction,
                )

                # Create batch entry for selected paper
                paper_entry = {
                    "custom_id": f"structured-rep-rel-{submission_id}-{paper_id}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model_name,
                        "messages": [{"role": "user", "content": paper_prompt}],
                        "temperature": 0.0,
                        "response_format": {"type": "json_schema", "json_schema": sch},
                    },
                }

                batch_entries.append(paper_entry)

        # Report introduction statistics
        print(
            f"Main papers with introductions: {main_papers_with_intro}/{total_main_papers} ({main_papers_with_intro/total_main_papers*100:.1f}%)"
        )
        print(
            f"Selected papers with introductions: {selected_papers_with_intro}/{total_selected_papers} ({selected_papers_with_intro/total_selected_papers*100:.1f}%)"
        )

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Write to JSONL file
        output_path = os.path.join(output_dir, "batch_structured_rep_input.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in batch_entries:
                f.write(json.dumps(entry) + "\n")

        print(
            f"Batch inference data saved to {output_path} with {len(batch_entries)} total entries"
        )
        return output_path

    def process_batch_results(
        self,
        results_file: str,
        data_dir: str,
    ) -> Dict[str, Any]:
        """Processes the output from an OpenAI batch API job for structured extraction.

        This method reads the JSONL results file, groups the extracted information
        by submission ID, and saves the complete structured representation for each
        submission to a JSON file. It also compiles and returns statistics about
        the batch job.

        Args:
            results_file: The path to the JSONL file containing the batch API results.
            data_dir: The base directory where the output for each submission
                      will be stored.

        Returns:
            A dictionary containing detailed statistics about the processed batch.
        """
        from litellm import cost_per_token

        # Load the results
        with open(results_file, "r", encoding="utf-8") as f:
            results = [json.loads(line) for line in f]

        # Group results by submission_id
        grouped_results = {}

        for result in results:
            custom_id = result.get("custom_id", "")

            if not custom_id.startswith("structured-rep-"):
                continue

            parts = custom_id.split("-")
            if len(parts) < 4:
                continue

            paper_type = parts[2]  # "main" or "rel"
            submission_id = parts[3]

            if submission_id not in grouped_results:
                grouped_results[submission_id] = {
                    "main_paper": None,
                    "selected_papers": {},
                }

            response_data = result.get("response", {}).get("body", {})
            choices = response_data.get("choices", [])

            if not choices:
                print(f"No content found for {custom_id}")
                continue

            content = choices[0].get("message", {}).get("content", "{}")
            usage = response_data.get("usage", {})

            # Calculate cost
            total_cost = sum(
                cost_per_token(
                    model="gpt-4.1",
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                )
            )

            # Parse JSON content
            try:
                parsed_content = json.loads(content)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON for {custom_id}")
                continue

            result_data = {
                "raw": content,
                "parsed": parsed_content,
                "token_usage": usage,
                "total_cost": total_cost,
            }

            if paper_type == "main":
                grouped_results[submission_id]["main_paper"] = result_data
            elif paper_type == "rel":
                rel_idx = (
                    parts[4]
                    if len(parts) > 4
                    else len(grouped_results[submission_id]["selected_papers"])
                )

                # Make sure we have enough slots
                # while len(grouped_results[submission_id]["selected_papers"]) <= rel_idx:
                #     grouped_results[submission_id]["selected_papers"].append(None)

                grouped_results[submission_id]["selected_papers"][rel_idx] = result_data

        # Statistics
        stats = {"total_processed": 0, "successful": 0, "failed": 0, "submissions": []}

        submissions = os.listdir(data_dir)
        for submission in submissions:
            submission_id = submission

            if submission_id not in grouped_results:
                print(f"No results found for {submission_id}")
                stats["failed"] += 1
                continue

            result_data = grouped_results[submission_id]

            # Check if we have main paper results
            if not result_data.get("main_paper"):
                print(f"No main paper results for {submission_id}")
                stats["failed"] += 1
                continue

            stats["total_processed"] += 1

            structured_representation = {
                "main_paper": {
                    "raw": result_data["main_paper"]["raw"],
                    "parsed": result_data["main_paper"]["parsed"],
                },
                "selected_papers": [
                    {
                        "paper_id": paper_k,
                        "raw": paper_v["raw"],
                        "parsed": paper_v["parsed"],
                    }
                    for paper_k, paper_v in result_data["selected_papers"].items()
                ],
            }

            # Save metadata
            meta_path = os.path.join(data_dir, submission_id, "ours", "metadata.json")
            current_meta = {}
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    current_meta = json.load(f)

            current_meta["structured_generation"] = {
                "main_paper": {
                    "total_cost": result_data["main_paper"]["total_cost"],
                    "token_usage": result_data["main_paper"]["token_usage"],
                },
                "selected_papers": [
                    {
                        "total_cost": paper_v["total_cost"],
                        "token_usage": paper_v["token_usage"],
                    }
                    for paper_k, paper_v in result_data["selected_papers"].items()
                ],
            }

            with open(meta_path, "w") as f:
                json.dump(current_meta, f, indent=3)

            # Save structured representation
            with open(
                f"{data_dir}/{submission_id}/structured_representation.json", "w"
            ) as f:
                json.dump(
                    {"structured_representation": structured_representation},
                    f,
                    indent=4,
                )

            print(f"Processed results for {submission_id}")
            stats["successful"] += 1
            stats["submissions"].append(submission_id)

        # Print summary
        print(f"\nProcessing Summary:")
        print(f"Total processed: {stats['total_processed']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")

        return stats


if __name__ == "__main__":
    import argparse
    """The main entry point for the script.

    Provides a command-line interface for structured data extraction. It supports
    two main operations:
    1. `--prepare`: Generates a JSONL file for batch processing with the OpenAI API.
    2. `--process`: Processes the results from a completed OpenAI batch job.
    """

    parser = argparse.ArgumentParser(
        description="Generate structured representations of papers"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../../novelty_assessment/data_iclr_all_topics/iclr_compiled_v2",
        help="Directory containing submission data",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4.1", help="OpenAI model to use"
    )
    parser.add_argument(
        "--prepare", action="store_true", help="Prepare batch inference data"
    )
    parser.add_argument("--process", action="store_true", help="Process batch results")
    parser.add_argument(
        "--results-file",
        type=str,
        default="./openai_outputs/batch_structured_rep_output.jsonl",
        help="Path to batch results file",
    )

    args = parser.parse_args()

    structured_extraction = StructuredRepresentationGenerator(model_name=args.model)

    if args.prepare:
        # Step 1: Prepare batch inference data
        structured_extraction.prepare_batch_inference_data(
            data_dir=args.data_dir,
            model_name=args.model,
        )

    if args.process:
        # Step 2: Process batch results
        structured_extraction.process_batch_results(
            results_file=args.results_file,
            data_dir=args.data_dir,
        )
