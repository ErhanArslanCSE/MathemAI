# Data Schema Documentation

This document provides detailed information about the data structures used in the MathemAI project. Understanding these schemas is essential for developers working with the data or extending the models.

## Overview

MathemAI uses three primary datasets:

1. **Dyscalculia Assessment Data**: Contains student profiles and diagnostic information
2. **Error Analysis Data**: Records specific mathematical errors and their patterns
3. **Intervention Tracking Data**: Tracks interventions and their effectiveness

## Dyscalculia Assessment Data

This dataset contains baseline assessment information for each student, including math skills, cognitive measures, and diagnostic information.

### Schema: `dyscalculia_assessment_data.csv`

| Field Name | Type | Description | Possible Values |
|------------|------|-------------|-----------------|
| student_id | Integer | Unique identifier for each student | Positive integers |
| age | Integer | Student's age in years | 6-12 |
| grade | Integer | Student's grade level | 1-7 |
| number_recognition | Integer | Ability to recognize and identify numbers | 1-5 (1=low, 5=high) |
| number_comparison | Integer | Ability to compare numerical values | 1-5 (1=low, 5=high) |
| counting_skills | Integer | Proficiency in counting sequences | 1-5 (1=low, 5=high) |
| place_value | Integer | Understanding of place value concepts | 1-5 (1=low, 5=high) |
| calculation_accuracy | Integer | Accuracy in performing calculations | 1-5 (1=low, 5=high) |
| calculation_fluency | Integer | Speed and efficiency in calculations | 1-5 (1=low, 5=high) |
| arithmetic_facts_recall | Integer | Ability to recall basic math facts | 1-5 (1=low, 5=high) |
| word_problem_solving | Integer | Ability to solve mathematical word problems | 1-5 (1=low, 5=high) |
| math_anxiety_level | String | Level of anxiety related to mathematics | "low", "medium", "high" |
| attention_score | String | Assessment of attention capabilities | "normal", "low", "very_low" |
| working_memory_score | String | Assessment of working memory | "normal", "low", "very_low" |
| visual_spatial_score | String | Assessment of visual-spatial abilities | "normal", "low", "very_low" |
| error_patterns | String | Predominant type of errors observed | "transposition", "reversal", "miscounting", "sequence_error", "operation_confusion", "multiple_errors", "calculation_error", "occasional_error", "consistent_error", "rare_error", "none" |
| response_time | String | Typical response time on math tasks | "fast", "average", "slow", "very_slow" |
| diagnosis | String | Diagnostic category based on assessment | "dyscalculia", "math_difficulty", "typical" |

### Derived Values

When processing this data:

- **Dyscalculia Risk Score**: Can be calculated as a weighted sum of the assessment scores, with lower scores indicating higher risk
- **Cognitive Profile**: Can be derived from the combination of working memory, attention, and visual-spatial scores
- **Severity Level**: Can be categorized as mild, moderate, or severe based on overall scores

## Error Analysis Data

This dataset captures specific mathematical errors made by students, allowing for detailed pattern analysis.

### Schema: `error_analysis_data.csv`

| Field Name | Type | Description | Possible Values |
|------------|------|-------------|-----------------|
| student_id | Integer | Unique identifier for each student | Matches student_id in assessment data |
| question_id | String | Unique identifier for each question | Format: "Q" followed by a number |
| question_type | String | Category of mathematical question | "number_recognition", "number_comparison", "counting", "number_sequence", "addition", "subtraction", "multiplication", "division", "place_value", "fractions", "word_problem" |
| question | String | The actual question presented to the student | Text of the question |
| student_answer | String | The answer provided by the student | Varies based on question |
| correct_answer | String | The correct answer to the question | Varies based on question |
| is_correct | Integer | Whether the student's answer was correct | 0 (incorrect) or 1 (correct) |
| response_time_seconds | Float | Time taken to respond in seconds | Positive float values |
| attempt_count | Integer | Number of attempts made before final answer | Positive integers |
| session_date | String | Date when the question was answered | Format: "YYYY-MM-DD" |

### Special Considerations

- **Multiple Attempts**: When a student makes multiple attempts at a question, only the final attempt is recorded in the main data, but the attempt_count field indicates how many tries were made
- **Response Time**: Longer response times often correlate with specific types of difficulties
- **Session Grouping**: Questions from the same assessment session share the same session_date, allowing for session-level analysis

## Intervention Tracking Data

This dataset tracks intervention strategies and their effectiveness for students with dyscalculia or math difficulties.

### Schema: `intervention_tracking_data.csv`

| Field Name | Type | Description | Possible Values |
|------------|------|-------------|-----------------|
| student_id | Integer | Unique identifier for each student | Matches student_id in assessment data |
| intervention_id | String | Unique identifier for each intervention | Format: "INT" followed by a 3-digit number |
| intervention_type | String | The type of intervention used | "multisensory_approach", "visual_aids", "game_based_learning", "structured_sequence", "technology_assisted" |
| start_date | String | Date when intervention began | Format: "YYYY-MM-DD" |
| end_date | String | Date when intervention ended | Format: "YYYY-MM-DD" |
| duration_weeks | Integer | Duration of the intervention in weeks | Positive integers |
| sessions_completed | Integer | Number of intervention sessions completed | Positive integers |
| pre_assessment_score | Integer | Assessment score before intervention | 0-100 |
| post_assessment_score | Integer | Assessment score after intervention | 0-100 |
| number_recognition_improvement | Integer | Improvement in number recognition | 0-3 typical range |
| number_comparison_improvement | Integer | Improvement in number comparison | 0-3 typical range |
| counting_improvement | Integer | Improvement in counting skills | 0-3 typical range |
| calculation_improvement | Integer | Improvement in calculation abilities | 0-3 typical range |
| problem_solving_improvement | Integer | Improvement in problem-solving | 0-3 typical range |
| math_anxiety_change | String | Change in math anxiety levels | "decreased", "no_change", "increased" |
| teacher_feedback | String | Qualitative feedback from teachers | Text feedback |
| parent_feedback | String | Qualitative feedback from parents | Text feedback |

### Effectiveness Metrics

When analyzing intervention data:

- **Overall Improvement**: Calculated as post_assessment_score - pre_assessment_score
- **Normalized Gain**: Calculated as (post_score - pre_score) / (100 - pre_score) to account for different starting points
- **Per-Skill Improvement**: Individual improvement metrics for specific math skills
- **Anxiety Reduction**: Changes in math anxiety levels as a non-cognitive outcome measure

## Data Relationships

The three datasets are linked through the `student_id` field, allowing for comprehensive analysis of a student's profile, errors, and intervention outcomes.

### Entity Relationship Diagram

```
+------------------------+       +------------------------+       +------------------------+
|                        |       |                        |       |                        |
| Assessment Data        |       | Error Analysis Data    |       | Intervention Data      |
|                        |       |                        |       |                        |
| student_id (PK)        |<----->| student_id (FK)        |<----->| student_id (FK)        |
| age                    |       | question_id            |       | intervention_id (PK)   |
| grade                  |       | question_type          |       | intervention_type      |
| skill_metrics...       |       | error_details...       |       | effectiveness_metrics...|
| diagnosis              |       |                        |       |                        |
|                        |       |                        |       |                        |
+------------------------+       +------------------------+       +------------------------+
```

## Data Quality Considerations

### Value Constraints

- Numeric assessment scores should be between 1-5
- Student ages should be within typical school-age range (6-12)
- Improvement metrics typically range from 0-3
- Pre/post assessment scores range from 0-100

### Missing Values

- In the assessment data, missing values for cognitive measures (working_memory_score, etc.) should be rare but may occur if assessments were incomplete
- In the error analysis data, attempt_count may be missing for some early records
- In the intervention data, teacher_feedback or parent_feedback may be missing in some cases

### Data Transformations

When preparing data for model training:

1. **Categorical Encoding**: Convert string values to numeric
   - For math_anxiety_level: "low"=0, "medium"=1, "high"=2
   - For attention_score/working_memory_score/visual_spatial_score: "normal"=2, "low"=1, "very_low"=0

2. **Feature Scaling**: Standardize numeric features to have mean=0 and std=1
   - Particularly important for the assessment scores before model training

3. **Temporal Features**: Extract additional features from dates
   - Intervention duration can be calculated from start_date and end_date
   - Session frequency can be derived from sessions_completed and duration_weeks

## Data Collection Guidelines

For those contributing new data to the project:

1. **Assessment Data**:
   - Ensure all skill assessments use the standardized 1-5 scale
   - Cognitive measures should be determined by qualified educators or specialists
   - Diagnosis should follow established criteria for dyscalculia and math difficulties

2. **Error Analysis Data**:
   - Record questions exactly as presented to the student
   - Measure response times consistently
   - Document all attempts, even if only the final attempt is recorded in the main data

3. **Intervention Data**:
   - Use consistent pre/post assessment methods
   - Record qualitative feedback using open-ended questions
   - Document the specific components of each intervention type

## Using the Data with MathemAI Models

### Screening Model

The screening model primarily uses the assessment data, with the following fields being most important:

- All skill metrics (number_recognition, calculation_accuracy, etc.)
- Cognitive measures (working_memory_score, visual_spatial_score, attention_score)
- Error patterns and response time

### Intervention Recommender

The intervention recommender uses both assessment data and historical intervention data to make recommendations. Key fields include:

- Student profile from assessment data
- Historical effectiveness of different interventions for similar profiles
- Specific areas of difficulty (based on low scores in particular skills)

## Data Privacy and Ethical Considerations

All data used in the MathemAI project must adhere to strict privacy standards:

1. **Anonymization**: All student_id values should be anonymized and not mappable to actual student identities
2. **Aggregation**: Public analyses should use aggregated data rather than individual records
3. **Minimal Collection**: Only collect data necessary for the project's objectives
4. **Informed Consent**: Ensure appropriate consent is obtained before collecting real student data
5. **Secure Storage**: Follow best practices for secure data storage and handling

## Data Versioning

As the project evolves, the data schemas may be updated. Version changes are documented as follows:

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-05-01 | Initial schema definitions |
| 1.1 | 2025-05-15 | Added error_patterns field to assessment data |

## Sample Data

Sample records from each dataset are provided below for reference:

### Sample Assessment Record

```json
{
  "student_id": 12,
  "age": 8,
  "grade": 3,
  "number_recognition": 3,
  "number_comparison": 4,
  "counting_skills": 5,
  "place_value": 3,
  "calculation_accuracy": 2,
  "calculation_fluency": 2,
  "arithmetic_facts_recall": 3,
  "word_problem_solving": 2,
  "math_anxiety_level": "high",
  "attention_score": "normal",
  "working_memory_score": "low",
  "visual_spatial_score": "normal",
  "error_patterns": "calculation_error",
  "response_time": "slow",
  "diagnosis": "math_difficulty"
}
```

### Sample Error Analysis Record

```json
{
  "student_id": 12,
  "question_id": "Q45",
  "question_type": "addition",
  "question": "9 + 6 = ?",
  "student_answer": "14",
  "correct_answer": "15",
  "is_correct": 0,
  "response_time_seconds": 17.3,
  "attempt_count": 2,
  "session_date": "2024-09-15"
}
```

### Sample Intervention Record

```json
{
  "student_id": 12,
  "intervention_id": "INT008",
  "intervention_type": "visual_aids",
  "start_date": "2024-09-18",
  "end_date": "2024-10-30",
  "duration_weeks": 6,
  "sessions_completed": 10,
  "pre_assessment_score": 45,
  "post_assessment_score": 58,
  "number_recognition_improvement": 1,
  "number_comparison_improvement": 1,
  "counting_improvement": 0,
  "calculation_improvement": 2,
  "problem_solving_improvement": 1,
  "math_anxiety_change": "decreased",
  "teacher_feedback": "Visual number lines have been particularly helpful",
  "parent_feedback": "Less frustration when practicing"
}
```