# MathemAI: AI-Assisted Learning for Dyscalculia and Math Difficulties

![MathemAI Logo](docs/figures/mathemat_logo.png)

MathemAI is an open-source project dedicated to leveraging artificial intelligence to help students with dyscalculia and other mathematics learning difficulties. The project aims to provide personalized learning recommendations, track progress, and deliver adaptive interventions tailored to each student's unique learning profile.

## Project Vision

Dyscalculia affects approximately 5-8% of school-age children, making it as common as dyslexia, yet it receives far less attention. MathemAI aims to address this gap by providing:

1. **Early Detection**: AI-powered screening to identify potential dyscalculia or math learning difficulties
2. **Personalized Interventions**: Custom learning paths based on specific error patterns and challenges
3. **Progress Tracking**: Detailed analytics to monitor improvements and adjust interventions
4. **Research Platform**: A data collection framework to advance our understanding of dyscalculia

## Features

- **Dyscalculia Screening Model**: Machine learning model that identifies patterns consistent with dyscalculia and math learning difficulties
- **Intervention Recommender**: Recommends personalized intervention strategies based on student profiles and past effectiveness
- **Error Pattern Analysis**: Identifies specific math error patterns to target interventions
- **RESTful API**: Integrates seamlessly with other educational platforms and applications
- **Data Collection Framework**: Ethically collects anonymized data to improve models and advance research

## Repository Structure

```
MathemAI/
├── api/                  # API implementation
│   └── app.py            # Flask-based API server
├── datasets/             # Data for training and testing
│   ├── dyscalculia_assessment_data.csv
│   ├── error_analysis_data.csv
│   └── intervention_tracking_data.csv
├── docs/                 # Documentation
│   └── figures/          # Images and visualizations
├── frontend/             # Web interface (to be implemented)
├── models/               # Model implementations
│   ├── dyscalculia_screening_model.py
│   └── intervention_recommender.py
├── notebooks/            # Jupyter notebooks for analysis
├── scripts/              # Utility scripts
│   ├── generate_datasets.py
│   └── train_models.py
├── tests/                # Test suite
├── .gitignore            # Git ignore file
├── LICENSE               # MIT License
├── README.md             # This file
└── requirements.txt      # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:

```bash
git clone https://github.com/openimpactai/OpenImpactAI.git
cd OpenImpactAI/AI-Education-Projects/MathemAI
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Generate sample datasets:

```bash
python scripts/generate_datasets.py
```

4. Train the models:

```bash
python scripts/train_models.py
```

5. Start the API server:

```bash
cd api
python app.py
```

The API will be available at `http://localhost:5000`.

## API Usage

### Screening for Dyscalculia

```bash
curl -X POST http://localhost:5000/api/screen \
  -H "Content-Type: application/json" \
  -d '{
    "number_recognition": 3,
    "number_comparison": 2,
    "counting_skills": 4,
    "place_value": 2,
    "calculation_accuracy": 2,
    "calculation_fluency": 1,
    "arithmetic_facts_recall": 2,
    "word_problem_solving": 1,
    "working_memory_score": "low",
    "visual_spatial_score": "normal",
    "math_anxiety_level": "high",
    "attention_score": "normal"
  }'
```

### Getting Intervention Recommendations

```bash
curl -X POST http://localhost:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "number_recognition": 3,
    "number_comparison": 2,
    "counting_skills": 4,
    "place_value": 2,
    "calculation_accuracy": 2,
    "calculation_fluency": 1,
    "arithmetic_facts_recall": 2,
    "word_problem_solving": 1,
    "working_memory_score": "low",
    "visual_spatial_score": "normal",
    "math_anxiety_level": "high",
    "attention_score": "normal"
  }'
```

## Data Description

### Dyscalculia Assessment Data

This dataset contains assessment data used to identify potential dyscalculia or math learning difficulties.

Key fields:
- `student_id`: Unique identifier for each student
- `diagnosis`: The diagnosis ('dyscalculia', 'math_difficulty', or 'typical')
- Various skill assessments (number recognition, calculation, etc.) on a scale of 1-5
- Cognitive measures (working memory, visual-spatial skills, etc.)

### Error Analysis Data

This dataset captures specific mathematical errors made by students.

Key fields:
- `student_id`: Unique identifier for each student
- `question_type`: Type of math question (addition, subtraction, counting, etc.)
- `question`: The actual math question presented
- `student_answer`: The answer provided by the student
- `correct_answer`: The correct answer
- `is_correct`: Whether the student's answer was correct (1) or incorrect (0)
- `response_time_seconds`: How long it took the student to respond
- `error_patterns`: Type of error made (transposition, reversal, etc.)

### Intervention Tracking Data

This dataset tracks interventions and their effectiveness.

Key fields:
- `student_id`: Unique identifier for each student
- `intervention_type`: Type of intervention (multisensory_approach, visual_aids, etc.)
- `pre_assessment_score` and `post_assessment_score`: Scores before and after intervention
- Various improvement metrics (number_recognition_improvement, etc.)
- `teacher_feedback` and `parent_feedback`: Qualitative feedback on intervention effectiveness

## Contributing

If you're interested in contributing to MathemAI, please check out our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines and opportunities.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.