import os
import json
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from rapidfuzz import fuzz
from dotenv import load_dotenv

# Load environment variables
load_dotenv()




# This prompt integrates adding citation contexts from paper for checking out the arguments mentioned by the authors.
novelty_delta_analysis_prompt = """
# Novelty Delta Analysis for Reviewer Support

## Task
Independently analyze how the submission paper's contributions relate to existing work in the field, critically examining both author claims and actual relationships. This analysis should help reviewers assess novelty by providing objective comparisons with prior work.

## Input Format
You will be provided with:
1. The structured information from the submission paper
2. A comprehensive research landscape analysis
3. Citation sentences for key related papers (how authors cite and characterize these works)

## Key Analysis Principles
- Independently verify relationships between submission and prior work
- Critically examine how authors characterize and compare with prior work
- Identify discrepancies between author characterizations and actual relationships
- Present evidence-based observations without making final judgments
- Distinguish between author-claimed differences and independently observed differences
- Provide context about field maturity and related work

## Output Format
Provide a detailed analysis with the following sections:

1. RESEARCH CONTEXT POSITIONING
   - Situate the submission within the identified research landscape
   - Identify the most closely related prior works
   - Independently assess how the submission relates to existing methodological clusters
   - Analyze its place within the problem space and evaluation approaches
   - Note: Do not accept author positioning claims without verification

2. AUTHOR CITATION ANALYSIS
   - Analyze how authors characterize and compare with each cited related work
   - Identify patterns in how authors position their contributions relative to others
   - Assess whether characterizations of prior work are accurate and balanced
   - Note discrepancies between how authors describe prior work and independent assessment
   - Evaluate whether claimed improvements or differences are substantiated
   - Identify rhetoric that may overstate differences or understate similarities

3. CONTRIBUTION DELTA ANALYSIS
   For each main contribution claimed in the submission:
   - Identify the most similar prior work for this specific contribution
   - Critically examine whether claimed differences actually exist
   - Detail exactly how this contribution differs from prior work, based on evidence
   - Compare author characterizations with independently verified relationships
   - Distinguish between substantive differences and superficial variations
   - Note when author claims about novelty or extension may be overstated
   - Consider whether improvements might be due to implementation details rather than conceptual advances
   - Note: Present factual observations about deltas without accepting author framing

4. FIELD CONTEXT CONSIDERATIONS
   - Provide information about how active/mature this research area is
   - Identify recent survey papers or literature reviews in this space
   - Note trends in how the field has been evolving
   - Present context about typical incremental advances in this field
   - Note: Offer context that helps reviewers calibrate their expectations

5. CRITICAL ASSESSMENT CONSIDERATIONS
   - Identify aspects where claimed novelty may be overstated
   - Analyze whether authors' characterizations of their own novelty align with evidence
   - Consider whether empirical improvements might result from factors other than claimed innovations
   - Assess whether terminology differences might mask conceptual similarities
   - Identify instances where "extensions" might be routine adaptations
   - Note: Frame these as considerations rather than definitive judgments

6. RELATED WORK CONSIDERATIONS
   - Identify potentially relevant work not addressed in the submission
   - Highlight areas where additional comparisons are necessary
   - Note incomplete or potentially misleading characterizations of prior work
   - Identify when claimed "limitations" of prior work may be exaggerated
   - Compare how authors cite specific works versus how they actually relate
   - Note: Present these as information that might help complete the picture

7. KEY OBSERVATION SUMMARY
   - Highlight the most significant independently verified differences from prior work
   - Summarize the main relationships to existing research
   - Identify which claimed contributions have the strongest and weakest differentiation
   - Note the most important discrepancies between author characterizations and independent assessment
   - Note: Frame as observations to inform the reviewer's independent judgment

## Evidence Standards
For each observation, provide:
- Specific references to prior work
- Clear distinction between author claims and independently verified differences
- Explicit identification of similarities and differences based on technical details
- Assessment of whether differences appear substantive or superficial
- Analysis of accuracy in how authors characterize related work

## Example Format for Citation Analysis
"For [Paper X], the authors characterize it as 'limited to simple datasets' and claim their work 'extends X to complex scenarios.' The citation sentences appear in the following contexts:
- 'Unlike X, which only works on simple datasets, our approach handles complex scenarios' (Introduction)
- 'X proposed the basic framework, but did not address challenge Y' (Related Work)

Independent analysis suggests that Paper X actually did address complex scenarios in Section 3.2, though using different terminology. The authors' characterization appears to understate X's capabilities to emphasize their contribution. The actual primary difference appears to be [specific technical difference] rather than the complexity of supported scenarios."

Remember that your role is to provide objective analysis that helps reviewers make informed judgments about novelty. Carefully examine both what authors explicitly claim and how they implicitly position their work through their characterizations of prior research.

{structured_representation}

## Papers from related work not cited
{not_cited_paper_titles}


##Citation Context
{citation_contexts}

## Research Landscape
{research_landscape}
"""




# Reuse PaperInformation model from landscape analysis
class PaperInformation(BaseModel):
    """A Pydantic model for storing structured information about a research paper.

    This model is used to ensure that the data extracted from papers (both the
    submission and related works) adheres to a consistent structure, which is
    then used for analysis and prompting.

    Attributes:
        paper_id: The unique identifier for the paper.
        title: The title of the paper.
        methods: A list of methods or approaches proposed in the paper.
        problems: A list of problems the paper addresses.
        datasets: A list of datasets used for evaluation.
        metrics: A list of evaluation metrics used.
        results: A list of key quantitative results, typically as dicts.
        novelty_claims: A list of explicit claims about the paper's novelty.
        is_submission: A boolean flag to indicate if this is the submission paper.
    """

    paper_id: str
    title: str
    methods: List[str] = Field(
        description="List of methods/approaches proposed in the paper"
    )
    problems: List[str] = Field(description="List of problems the paper addresses")
    datasets: List[str] = Field(description="List of datasets used for evaluation")
    metrics: List[str] = Field(description="List of evaluation metrics used")
    results: List[Dict[str, str]] = Field(
        description="Key quantitative results with metric name -> value pairs"
    )
    novelty_claims: List[str] = Field(
        description="Claims about what is novel in this work"
    )
    is_submission: bool = False


class NoveltyAssessor:
    """Performs a detailed novelty assessment of a submission paper.

    This class uses a large language model (LLM) to analyze a submission paper
    in the context of its cited works and a broader research landscape. It aims
    to provide an objective, evidence-based analysis to support peer reviewers.

    Attributes:
        llm: An instance of a LangChain ChatOpenAI model used for the assessment.
    """

    def __init__(self, model_name: str = "gpt-4.1", temperature: float = 0.0):
        """Initializes the NoveltyAssessor.

        Args:
            model_name: The identifier for the OpenAI model to be used.
            temperature: The temperature setting for the LLM, controlling the
                         randomness of the output.

        Raises:
            ValueError: If the `OPENAI_API_KEY` environment variable is not set.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            api_key=api_key,
        )

    @staticmethod
    def get_papers_not_cited(cited_papers: List[Dict], selected_papers: List[Dict], threshold: int = 85) -> List[Dict]:
        """Identifies papers from a selected list that are not in a cited list.

        This function uses fuzzy string matching on paper titles to determine if a
        paper from `selected_papers` is already present in `cited_papers`.

        Args:
            cited_papers: A list of dictionaries, each representing a cited paper.
                          Must contain a "title" key.
            selected_papers: A list of dictionaries representing papers to check.
                             Must contain a "title" key.
            threshold: The similarity score (0-100) from `rapidfuzz` above which
                       two titles are considered a match.

        Returns:
            A list of paper dictionaries from `selected_papers` that were not
            found in `cited_papers`.
        """
        not_cited_papers = []
        for paper in selected_papers:
            is_cited = False
            for cited in cited_papers:
                if "title" not in cited:
                    continue
                if fuzz.token_set_ratio(paper["title"], cited["title"]) >= threshold:
                    is_cited = True
                    break
            if not is_cited:
                not_cited_papers.append(paper)
        return not_cited_papers

    @staticmethod
    def get_citation_matches(citation_contexts: Dict[str, List[str]]) -> str:
        """Formats a dictionary of citation contexts into a readable string for the LLM prompt.

        Args:
            citation_contexts: A dictionary where keys are paper titles and values
                               are lists of sentences in which that paper was cited.

        Returns:
            A formatted string listing each paper and its citation contexts.
        """
        output_lines = []
        for title, context in citation_contexts.items():
            output_lines.append(f"📄 {title}\n")
            for ctx in context:
                output_lines.append(f"↳ Context: {ctx}\n")
        return "\n".join(output_lines)

    def load_submission_paper(
        self, data_dir: str, submission_id: str
    ) -> Optional[PaperInformation]:
        """Loads and parses the structured information for the main submission paper.

        Args:
            data_dir: The base directory where submission data is stored.
            submission_id: The unique identifier for the submission.

        Returns:
            A `PaperInformation` object for the submission paper, or None if the
            file cannot be found or parsed.

        Raises:
            ValueError: If the structured representation file is not found.
        """
        file_path = os.path.join(
            data_dir, submission_id, f"structured_representation.json"
        )

        if not os.path.exists(file_path):
            raise ValueError(f"Submission paper file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        with open(f"{data_dir}/{submission_id}/{submission_id}.json", "r") as f:
            paper_obj = json.load(f)

        structured_representation = data["structured_representation"]
        structured_representation["main_paper"]["parsed"]["paper_id"] = submission_id
        structured_representation["main_paper"]["parsed"]["is_submission"] = True
        structured_representation["main_paper"]["parsed"]["title"] = paper_obj["title"]
        paper_info = PaperInformation(
            **structured_representation["main_paper"]["parsed"]
        )

        return paper_info

    def load_citation_contexts(self, data_dir: str, submission_id: str) -> str:
        """Loads and formats the citation contexts for related papers.

        This method reads the main submission file and the ranked papers list to
        find the sentences where each ranked, cited paper was mentioned.

        Args:
            data_dir: The base directory for submission data.
            submission_id: The unique identifier for the submission.

        Returns:
            A formatted string of citation contexts suitable for the LLM prompt.
        """
        with open(f"{data_dir}/{submission_id}/{submission_id}.json", "r") as f:
            paper_obj = json.load(f)

        with open(
            f"{data_dir}/{submission_id}/related_work_data/ranked_papers.json", "r"
        ) as f:
            ranked_papers = json.load(f)

        title_and_sentences = {}

        for i in ranked_papers:
            ss_paper_id = i["paper_id"]
            citation_contexts = []
            for j in paper_obj["cited_papers"]:
                if "ss_paper_obj" not in j or j["ss_paper_obj"] is None:
                    continue
                if j["ss_paper_obj"]["paper_id"].strip() == ss_paper_id.strip():
                    if j["id"] not in [m[0] for m in citation_contexts]:
                        citation_contexts.append([j["id"], j["ss_paper_obj"]["title"]])

            for c in citation_contexts:
                title_and_sentences[c[1]] = [
                    k["context_sentence"]
                    for k in paper_obj["citation_contexts"]
                    if k["cited_paper_id"].strip() == c[0].strip()
                ]
                if len(title_and_sentences[c[1]]) == 0:
                    del title_and_sentences[c[1]]

        citation_contexts = self.get_citation_matches(title_and_sentences)

        return citation_contexts


    def load_landscape_analysis(self, landscape_file: str) -> str:
        """Loads the content of a research landscape analysis file.

        Args:
            landscape_file: The path to the text file containing the analysis.

        Returns:
            The content of the file as a string.

        Raises:
            ValueError: If the specified file does not exist.
        """
        if not os.path.exists(landscape_file):
            raise ValueError(f"Landscape analysis file not found: {landscape_file}")

        with open(landscape_file, "r", encoding="utf-8") as f:
            analysis = f.read()

        return analysis

    def format_submission_for_prompt(self, submission: PaperInformation) -> str:
        """Formats the structured information of the submission paper for the LLM prompt.

        Args:
            submission: A `PaperInformation` object for the submission paper.

        Returns:
            A markdown-formatted string summarizing the paper's key aspects.
        """
        formatted_text = f"# SUBMISSION PAPER: {submission.title}\n\n"

        formatted_text += (
            f"## METHODS:\n"
            + "\n".join([f"- {m}" for m in submission.methods])
            + "\n\n"
        )
        formatted_text += (
            f"## PROBLEMS:\n"
            + "\n".join([f"- {p}" for p in submission.problems])
            + "\n\n"
        )
        formatted_text += (
            f"## DATASETS:\n"
            + "\n".join([f"- {d}" for d in submission.datasets])
            + "\n\n"
        )
        formatted_text += (
            f"## METRICS:\n"
            + "\n".join([f"- {m}" for m in submission.metrics])
            + "\n\n"
        )
        formatted_text += (
            f"## RESULTS:\n"
            + "\n".join([f"- {json.dumps(r)}" for r in submission.results])
            + "\n\n"
        )
        formatted_text += (
            f"## NOVELTY CLAIMS:\n"
            + "\n".join([f"- {c}" for c in submission.novelty_claims])
            + "\n\n"
        )

        return formatted_text

    def assess_novelty(
        self,
        submission: PaperInformation,
        landscape_analysis: str,
        not_cited_paper_titles: str,
        citation_contexts: str,
    ) -> str:
        """Performs the core novelty assessment by querying the LLM.

        This method constructs a detailed prompt containing the submission's
        structured data, the research landscape, and citation contexts, then
        invokes the LLM to generate the novelty analysis.

        Args:
            submission: The `PaperInformation` object for the submission.
            landscape_analysis: The text of the research landscape analysis.
            not_cited_paper_titles: A formatted string of titles of relevant but
                                    uncited papers.
            citation_contexts: A formatted string of citation contexts.

        Returns:
            The novelty assessment text generated by the LLM.
        """
        # Format submission for the prompt
        formatted_submission = self.format_submission_for_prompt(submission)

        prompt = novelty_delta_analysis_prompt.format(
            structured_representation=formatted_submission,
            research_landscape=landscape_analysis,
            not_cited_paper_titles=not_cited_paper_titles,
            citation_contexts=citation_contexts,
        )

        # Run the assessment
        assessment = self.llm.invoke(prompt)

        return assessment

    def format_not_cited_papers(self, non_cited_papers: list) -> str:
        """Formats a list of uncited paper titles into a simple string list.

        Args:
            non_cited_papers: A list of paper title strings.

        Returns:
            A single string with each title prefixed by a hyphen and newline.
        """
        formatted_text = "\n-".join([i for i in non_cited_papers])
        return formatted_text

    def run_assessment(
        self,
        data_dir: str,
        submission_id: str,
    ) -> str:
        """Runs the end-to-end novelty assessment pipeline for a single submission.

        This method orchestrates the loading of all necessary data, calls the
        assessment logic, saves the output, and updates metadata with cost and
        token usage statistics.

        Args:
            data_dir: The base directory for all submission data.
            submission_id: The unique identifier for the submission to be processed.

        Returns:
            The novelty assessment as a `langchain_core.messages.ai.AIMessage` object.
        """
        # Load submission paper
        submission = self.load_submission_paper(data_dir, submission_id)

        # Load landscape analysis (standard path)
        landscape_path = os.path.join(data_dir, submission_id, "research_landscape.txt")
        landscape_analysis = self.load_landscape_analysis(landscape_path)

        # Load non-cited papers from ranked papers
        with open(f"{data_dir}/{submission_id}/related_work_data/ranked_papers.json", "r") as f:
            ranked_papers = json.load(f)
        not_cited_papers = [i["title"] for i in ranked_papers if not i["cited_paper"]]
        formatted_not_cited_papers = self.format_not_cited_papers(not_cited_papers)

        # Load citation contexts
        citation_contexts = self.load_citation_contexts(data_dir, submission_id)

        # Run assessment
        assessment = self.assess_novelty(
            submission,
            landscape_analysis,
            formatted_not_cited_papers,
            citation_contexts,
        )

        # Save assessment to file (standard path)
        output_path = os.path.join(data_dir, submission_id, "novelty_delta_analysis.txt")
        if assessment:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(assessment.content)

        print(f"Novelty assessment saved to novelty_delta_analysis.txt")

        from litellm import cost_per_token

        tok_used = assessment.response_metadata["token_usage"]
        total_cost = sum(
            cost_per_token(
                model="gpt-4.1",
                prompt_tokens=tok_used["prompt_tokens"],
                completion_tokens=tok_used["completion_tokens"],
            )
        )

        meta_path = os.path.join(data_dir, submission_id, "metadata.json")
        current_meta = json.load(open(meta_path, "r"))

        current_meta["novelty_assessment"] = {
            "total_cost": total_cost,
            "token_usage": tok_used,
        }

        with open(meta_path, "w") as f:
            json.dump(current_meta, f, indent=3)

        return assessment

    def prepare_batch_inference_data(
        self,
        data_dir: str,
        landscape_file: str = "research_landscape.txt",
        model_name: str = "gpt-4.1",
    ) -> str:
        """Prepares a JSONL file for batch processing with the OpenAI API.

        This method iterates through all submission directories, gathers the
        necessary inputs (structured representation, landscape analysis, etc.),
        constructs a prompt for each, and writes them to a JSONL file suitable
        for the OpenAI batch API.

        Args:
            data_dir: The directory containing all submission subdirectories.
            landscape_file: The filename of the research landscape analysis.
            model_name: The OpenAI model identifier to be included in the batch request.

        Returns:
            The path to the generated JSONL batch input file.
        """
        batch_entries = []

        submission_ids = os.listdir(data_dir)

        # Statistics for reporting
        total_submissions = len(submission_ids)
        valid_submissions = 0
        missing_landscape = 0
        missing_structured_rep = 0

        for submission_id in submission_ids:
            # Check if required files exist
            landscape_path = os.path.join(
                data_dir, submission_id, "ours", landscape_file
            )
            structured_rep_path = os.path.join(
                data_dir, submission_id, "ours", "structured_representation.json"
            )

            if not os.path.exists(
                f"{data_dir}/{submission_id}/related_work_data/ranked_papers.json"
            ):
                print(f"Missing abstract and introduction for {submission_id}")
                continue

            with open(
                f"{data_dir}/{submission_id}/related_work_data/ranked_papers.json",
                "r",
            ) as f:
                ranked_papers = json.load(f)

            not_cited_papers = [
                i["title"] for i in ranked_papers if not i["cited_paper"]
            ]

            if not os.path.exists(landscape_path):
                print(f"Missing landscape analysis for {submission_id}")
                missing_landscape += 1
                continue

            if not os.path.exists(structured_rep_path):
                print(f"Missing structured representation for {submission_id}")
                missing_structured_rep += 1
                continue

            # Load submission paper
            submission = self.load_submission_paper(data_dir, submission_id)

            # Load landscape analysis
            landscape_analysis = self.load_landscape_analysis(landscape_path)

            # Load citation context
            citation_contexts = self.load_citation_contexts(data_dir, submission_id)

            # Format submission for the prompt
            formatted_submission = self.format_submission_for_prompt(submission)

            # Get non-cited papers
            formatted_not_cited_papers = self.format_not_cited_papers(not_cited_papers)

            # Prepare the prompt
            prompt = novelty_delta_analysis_prompt.format(
                structured_representation=formatted_submission,
                research_landscape=landscape_analysis,
                not_cited_paper_titles=formatted_not_cited_papers,
                citation_contexts=citation_contexts,
            )

            # Create the batch entry
            batch_entry = {
                "custom_id": f"novelty-assessment-{submission_id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                },
            }

            batch_entries.append(batch_entry)
            valid_submissions += 1

        # Report statistics
        print(
            f"Valid submissions for novelty assessment: {valid_submissions}/{total_submissions} ({valid_submissions/total_submissions*100:.1f}%)"
        )
        if missing_landscape > 0:
            print(f"Submissions missing landscape analysis: {missing_landscape}")
        if missing_structured_rep > 0:
            print(
                f"Submissions missing structured representation: {missing_structured_rep}"
            )

        # Write to JSONL file
        output_path = os.path.join(
            "./openai_inputs", "batch_novelty_assessment_input.jsonl"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in batch_entries:
                f.write(json.dumps(entry) + "\n")

        print(
            f"Batch inference data saved to {output_path} with {len(batch_entries)} total entries"
        )
        return output_path

    def process_batch_results(
        self,
        data_dir: str,
        results_file: str,
        output_file: str = "novelty_delta_analysis.txt",
    ) -> Dict:
        """Processes the output file from an OpenAI batch API job for novelty assessment.

        This method parses the JSONL results file, saves the generated assessment
        for each submission, calculates costs, updates metadata, and compiles
        overall statistics for the batch run.

        Args:
            data_dir: The base directory for storing submission data.
            results_file: The path to the JSONL file containing the batch results.
            output_file: The name for the output file where the assessment will be saved.

        Returns:
            A dictionary containing detailed statistics about the processed batch.
        """
        from litellm import cost_per_token

        print(f"Processing novelty assessment batch results from {results_file}")

        # Statistics for reporting
        stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "total_cost": 0.0,
            "submissions": [],
        }

        # Load the results
        with open(results_file, "r", encoding="utf-8") as f:
            results = [json.loads(line) for line in f]

        # Process each result
        for result in results:
            custom_id = result.get("custom_id", "")

            if not custom_id.startswith("novelty-assessment-"):
                print(f"Skipping non-novelty-assessment result: {custom_id}")
                continue

            stats["total_processed"] += 1

            # Extract submission_id from custom_id
            submission_id = custom_id.replace("novelty-assessment-", "")

            # Extract the assessment content
            response_data = result.get("response", {}).get("body", {})
            choices = response_data.get("choices", [])

            submission_stats = {"submission_id": submission_id, "status": "success"}

            if not choices:
                print(f"No content found for {submission_id}")
                stats["failed"] += 1
                submission_stats["status"] = "failed"
                submission_stats["error"] = "No content in choices"
                stats["submissions"].append(submission_stats)
                continue

            # Get the content from the first choice
            content = choices[0].get("message", {}).get("content", "")

            if not content:
                print(f"Empty content for {submission_id}")
                stats["failed"] += 1
                submission_stats["status"] = "failed"
                submission_stats["error"] = "Empty content"
                stats["submissions"].append(submission_stats)
                continue

            # Save the content to a file
            submission_dir = os.path.join(data_dir, submission_id, "ours")
            os.makedirs(submission_dir, exist_ok=True)

            output_path = os.path.join(submission_dir, output_file)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

            print(f"Saved novelty assessment for {submission_id}")

            # Save metadata
            token_usage = response_data.get("usage", {})

            # Calculate cost
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)

            try:
                cost = sum(
                    cost_per_token(
                        model="gpt-4.1",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                )
                stats["total_cost"] += cost
                submission_stats["cost"] = cost
            except Exception as e:
                print(f"Error calculating cost for {submission_id}: {e}")
                submission_stats["cost_error"] = str(e)

            submission_stats["token_usage"] = token_usage

            # Update metadata file
            meta_path = os.path.join(submission_dir, "metadata.json")
            current_meta = {}

            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        current_meta = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error reading metadata file for {submission_id}")

            current_meta["novelty_assessment"] = {
                "total_cost": submission_stats.get("cost", 0),
                "token_usage": token_usage,
            }

            with open(meta_path, "w") as f:
                json.dump(current_meta, f, indent=3)

            stats["successful"] += 1
            stats["submissions"].append(submission_stats)

        # Create summary statistics
        success_rate = (
            (stats["successful"] / stats["total_processed"]) * 100
            if stats["total_processed"] > 0
            else 0
        )
        print(
            f"Processed {stats['total_processed']} submissions with {stats['successful']} successes ({success_rate:.1f}%)"
        )
        print(f"Total estimated cost: ${stats['total_cost']:.4f}")

        # Save summary
        summary_path = "./openai_outputs/novelty_assessment_processing_summary.json"
        with open(summary_path, "w") as f:
            json.dump(stats, f, indent=3)

        return stats


if __name__ == "__main__":
    import argparse
    
    """The main entry point for the script.

    Provides a command-line interface to run the novelty assessment process.
    Supports three main modes:
    1. Processing a single submission.
    2. Preparing a batch input file for the OpenAI API.
    3. Processing the results from a completed OpenAI API batch job.
    """
    parser = argparse.ArgumentParser(description="Assess novelty of research papers")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../../novelty_assessment/data_iclr_all_topics/iclr_compiled_v2",
        help="Directory containing submission data"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1",
        help="OpenAI model to use"
    )
    parser.add_argument(
        "--submission-id",
        type=str,
        help="Process single submission (for run_assessment)"
    )
    parser.add_argument(
        "--landscape-file",
        type=str,
        default="research_landscape.txt",
        help="Path to landscape analysis file"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="novelty_delta_analysis.txt",
        help="Output file for assessment"
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare batch inference data"
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help="Process batch results"
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="./openai_outputs/batch_novelty_assessment_output.jsonl",
        help="Path to batch results file"
    )
    
    args = parser.parse_args()
    
    assessor = NoveltyAssessor(model_name=args.model)
    
    if args.submission_id:
        # Single submission assessment
        assessor.run_assessment(
            args.data_dir,
            args.submission_id
        )
    
    if args.prepare:
        # Prepare batch inference data
        assessor.prepare_batch_inference_data(args.data_dir)
    
    if args.process:
        # Process batch results
        assessor.process_batch_results(
            args.data_dir,
            args.results_file,
            "novelty_delta_analysis.txt"
        )
