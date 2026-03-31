# src/evaluation/testset.py

# This is our golden test set — 20 questions across the filings
# we downloaded. Ground truth answers are written by us after
# manually reading the filings.
#
# Why manual? Because RAGAS needs reliable ground truth to compute
# context recall. Auto-generated test sets exist but introduce noise.
# For a portfolio project, 20 high-quality manual pairs is better
# than 100 noisy auto-generated ones.

# src/evaluation/testset.py

TEST_SET = [
    # ── Apple 2025 (FY ended September 27, 2025) ──────────────────
    {
        "question": "What was Apple's total net sales in fiscal year 2025?",
        "ground_truth": "Apple's total net sales in fiscal year 2025 were $391.0 billion.",
    },
    {
        "question": "What was Apple's net income in fiscal year 2025?",
        "ground_truth": "Apple's net income in fiscal year 2025 was approximately $93.7 billion.",
    },
    {
        "question": "What were Apple's main revenue segments in 2025?",
        "ground_truth": "Apple's main revenue segments were iPhone, Mac, iPad, Wearables Home and Accessories, and Services.",
    },
    {
        "question": "When did Apple's fiscal year 2025 end?",
        "ground_truth": "Apple's fiscal year 2025 ended on September 27, 2025.",
    },
    {
        "question": "Where is Apple's principal executive office located?",
        "ground_truth": "Apple's principal executive office is located at One Apple Park Way, Cupertino, California 95014.",
    },
    {
        "question": "On which exchange is Apple stock listed?",
        "ground_truth": "Apple common stock is listed on The Nasdaq Stock Market LLC under the ticker symbol AAPL.",
    },

    # ── Amazon 2025 (FY ended December 31, 2025) ──────────────────
    {
        "question": "What was Amazon's total revenue in fiscal year 2025?",
        "ground_truth": "Amazon's total net sales in fiscal year 2025 were approximately $638 billion.",
    },
    {
        "question": "When did Amazon's fiscal year 2025 end?",
        "ground_truth": "Amazon's fiscal year 2025 ended on December 31, 2025.",
    },
    {
        "question": "Where is Amazon's principal executive office located?",
        "ground_truth": "Amazon's principal executive office is located at 410 Terry Avenue North, Seattle, Washington 98109.",
    },
    {
        "question": "On which exchange is Amazon stock listed?",
        "ground_truth": "Amazon common stock is listed on the Nasdaq Global Select Market under the ticker symbol AMZN.",
    },
    {
        "question": "What is Amazon's state of incorporation?",
        "ground_truth": "Amazon is incorporated in Delaware.",
    },

    # ── JPMorgan Chase 2025 (FY ended December 31, 2025) ──────────
    {
        "question": "What was JPMorgan Chase's net revenue in 2025?",
        "ground_truth": "JPMorgan Chase's total net revenue in 2025 was approximately $175 billion.",
    },
    {
        "question": "When did JPMorgan Chase's fiscal year 2025 end?",
        "ground_truth": "JPMorgan Chase's fiscal year 2025 ended on December 31, 2025.",
    },
    {
        "question": "Where is JPMorgan Chase's principal executive office?",
        "ground_truth": "JPMorgan Chase's principal executive office is located at 270 Park Avenue, New York, New York 10017.",
    },
    {
        "question": "On which exchange is JPMorgan Chase stock listed?",
        "ground_truth": "JPMorgan Chase common stock is listed on the New York Stock Exchange under the ticker symbol JPM.",
    },
    {
        "question": "What is JPMorgan Chase's state of incorporation?",
        "ground_truth": "JPMorgan Chase & Co. is incorporated in Delaware.",
    },
    {
        "question": "What is JPMorgan Chase's IRS employer identification number?",
        "ground_truth": "JPMorgan Chase's IRS employer identification number is 13-2624428.",
    },

    # ── Cross-company comparison questions ────────────────────────
    {
        "question": "Which companies have filed 10-K reports for fiscal year 2025?",
        "ground_truth": "Apple Inc., Amazon.com Inc., and JPMorgan Chase & Co. have all filed 10-K annual reports for fiscal year 2025.",
    },
    {
        "question": "Which of these companies are incorporated in Delaware?",
        "ground_truth": "Both Amazon and JPMorgan Chase are incorporated in Delaware. Apple is incorporated in California.",
    },
    {
        "question": "What commission file number does Apple have with the SEC?",
        "ground_truth": "Apple's SEC commission file number is 001-36743.",
    },
]