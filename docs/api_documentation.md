# API Documentation

This document provides detailed information about the MathemAI REST API, which allows integration with other applications and services.

## Base URL

All API endpoints are relative to the base URL:

```
http://localhost:5000
```

When deployed to production, the base URL will change accordingly.

## Authentication

Currently, the API uses a simple API key for certain restricted endpoints. Include the key in the request header:

```
X-API-Key: your_api_key_here
```

API keys can be configured in the server environment.

## Endpoints

### Health Check

Check if the API is running and the models are loaded.

**Endpoint:** GET `/health`

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "timestamp": "2025-05-20T12:34:56.789Z"
}
```

### Screening for Dyscalculia

Screen a student for dyscalculia and math learning difficulties based on assessment data.

**Endpoint:** POST `/api/screen`

**Request Body:**
```json
{
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
}
```

**Response:**
```json
{
  "screening_result": {
    "prediction": "dyscalculia",
    "confidence": 0.86,
    "probabilities": {
      "dyscalculia": 0.86,
      "math_difficulty": 0.12,
      "typical": 0.02
    }
  },
  "intervention_recommendation": {
    "cluster": 2,
    "recommended_interventions": [
      "multisensory_approach",
      "visual_aids",
      "structured_sequence"
    ],
    "description": "Based on the assessment profile, this student shows patterns similar to other students in Cluster 2. The recommended interventions are:\n\n- Multisensory Approach: Multisensory approaches use tactile and visual methods to help students understand mathematical concepts through multiple senses.\n\n- Visual Aids: Visual aids like number lines and manipulatives help students visualize mathematical relationships and build stronger conceptual understanding.\n\n"
  }
}
```

### Recommending Interventions

Get personalized intervention recommendations based on assessment data.

**Endpoint:** POST `/api/recommend`

**Request Body:**
```json
{
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
}
```

**Response:**
```json
{
  "cluster": 2,
  "recommended_interventions": [
    "multisensory_approach",
    "visual_aids",
    "structured_sequence"
  ],
  "description": "Based on the assessment profile, this student shows patterns similar to other students in Cluster 2. The recommended interventions are:\n\n- Multisensory Approach: Multisensory approaches use tactile and visual methods to help students understand mathematical concepts through multiple senses.\n\n- Visual Aids: Visual aids like number lines and manipulatives help students visualize mathematical relationships and build stronger conceptual understanding.\n\n"
}
```

### Saving Assessment Data

Save a new assessment record to the dataset.

**Endpoint:** POST `/api/save-assessment`

**Request Body:**
```json
{
  "age": 8,
  "grade": 3,
  "number_recognition": 3,
  "number_comparison": 2,
  "counting_skills": 4,
  "place_value": 2,
  "calculation_accuracy": 2,
  "calculation_fluency": 1,
  "arithmetic_facts_recall": 2,
  "word_problem_solving": 1,
  "math_anxiety_level": "high",
  "attention_score": "normal",
  "working_memory_score": "low",
  "visual_spatial_score": "normal",
  "error_patterns": "transposition",
  "response_time": "slow",
  "diagnosis": "dyscalculia"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Assessment data saved successfully",
  "student_id": 31
}
```

### Saving Intervention Data

Save a new intervention record to track effectiveness.

**Endpoint:** POST `/api/save-intervention`

**Request Body:**
```json
{
  "student_id": 12,
  "intervention_type": "multisensory_approach",
  "start_date": "2025-05-01",
  "end_date": "2025-06-12",
  "duration_weeks": 6,
  "sessions_completed": 10,
  "pre_assessment_score": 42,
  "post_assessment_score": 56,
  "number_recognition_improvement": 2,
  "number_comparison_improvement": 1,
  "counting_improvement": 1,
  "calculation_improvement": 2,
  "problem_solving_improvement": 1,
  "math_anxiety_change": "decreased",
  "teacher_feedback": "Student shows improved number recognition but still struggles with calculations",
  "parent_feedback": "Child is less anxious about math homework"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Intervention data saved successfully",
  "intervention_id": "INT026"
}
```

### Recording Error Patterns

Save detailed information about specific mathematical errors.

**Endpoint:** POST `/api/error-patterns`

**Request Body:**
```json
{
  "student_id": 15,
  "question_type": "addition",
  "question": "6 + 7 = ?",
  "student_answer": "12",
  "correct_answer": "13",
  "is_correct": 0,
  "response_time_seconds": 15.3,
  "attempt_count": 2
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Error pattern data saved successfully",
  "question_id": "Q64"
}
```

### Retraining Models

Trigger retraining of the models with the latest data. Requires API key authentication.

**Endpoint:** POST `/api/retrain`

**Headers:**
```
X-API-Key: your_api_key_here
```

**Response:**
```json
{
  "success": true,
  "timestamp": "2025-05-20T14:23:45.678Z",
  "models": {
    "screening": {
      "success": true,
      "model": "screening",
      "accuracy": 0.89,
      "timestamp": "2025-05-20T14:23:40.123Z"
    },
    "recommender": {
      "success": true,
      "model": "recommender",
      "n_clusters": 5,
      "timestamp": "2025-05-20T14:23:44.456Z"
    }
  }
}
```

### Exporting Data

Export datasets for research or backup. Requires API key authentication.

**Endpoint:** GET `/api/export-data?type=all&format=json`

**Parameters:**
- `type`: Which dataset to export (`all`, `assessment`, `intervention`, or `error`)
- `format`: Export format (currently only `json` is supported)

**Headers:**
```
X-API-Key: your_api_key_here
```

**Response:**
```json
{
  "assessment_data": [
    {
      "student_id": 1,
      "age": 7,
      "grade": 2,
      "number_recognition": 3,
      "number_comparison": 2,
      "counting_skills": 4,
      "place_value": 2,
      "calculation_accuracy": 2,
      "calculation_fluency": 1,
      "arithmetic_facts_recall": 2,
      "word_problem_solving": 1,
      "math_anxiety_level": "high",
      "attention_score": "normal",
      "working_memory_score": "low",
      "visual_spatial_score": "normal",
      "error_patterns": "transposition",
      "response_time": "slow",
      "diagnosis": "dyscalculia"
    },
    // ...more records
  ],
  "intervention_data": [
    // intervention records
  ],
  "error_data": [
    // error pattern records
  ]
}
```

### Getting Statistics

Get summary statistics about the collected data.

**Endpoint:** GET `/api/stats`

**Response:**
```json
{
  "timestamp": "2025-05-20T15:30:45.123Z",
  "datasets": {
    "assessment": {
      "total_records": 30,
      "dyscalculia_count": 12,
      "math_difficulty_count": 9,
      "typical_count": 9
    },
    "intervention": {
      "total_records": 25,
      "unique_students": 15,
      "intervention_types": {
        "multisensory_approach": 8,
        "visual_aids": 6,
        "game_based_learning": 5,
        "structured_sequence": 4,
        "technology_assisted": 2
      },
      "avg_improvement": 6.3
    },
    "error_analysis": {
      "total_records": 150,
      "unique_students": 18,
      "question_types": {
        "addition": 35,
        "subtraction": 32,
        "number_comparison": 28,
        "place_value": 25,
        "counting": 20,
        "word_problem": 10
      },
      "correct_percentage": 32.5,
      "avg_response_time": 14.7
    }
  }
}
```

## Error Responses

The API returns appropriate HTTP status codes for different types of errors:

- **400 Bad Request**: Invalid input data
- **401 Unauthorized**: Missing or invalid API key
- **404 Not Found**: Resource not found
- **500 Internal Server Error**: Server-side error

Example error response:

```json
{
  "error": "An error occurred during screening: Invalid assessment data"
}
```

## Client-Side Error Logging

Report client-side errors to the server.

**Endpoint:** POST `/api/log-error`

**Request Body:**
```json
{
  "message": "Error description",
  "stack": "Error stack trace",
  "browser": "Chrome 113.0.0.0",
  "os": "Windows 10"
}
```

**Response:**
```json
{
  "status": "error logged"
}
```

## Rate Limiting

To ensure fair usage, the API implements rate limiting:

- 100 requests per minute for public endpoints
- 300 requests per minute for authenticated endpoints

When rate limits are exceeded, the API returns a 429 Too Many Requests status code.

## Versioning

The current API version is v1. All endpoints should be prefixed with `/v1` when version control is implemented.

## Integration Examples

### Python Example

```python
import requests
import json

base_url = "http://localhost:5000"

# Screening for dyscalculia
assessment_data = {
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
}

response = requests.post(
    f"{base_url}/api/screen",
    headers={"Content-Type": "application/json"},
    data=json.dumps(assessment_data)
)

if response.status_code == 200:
    result = response.json()
    print(f"Screening result: {result['screening_result']['prediction']}")
    print(f"Confidence: {result['screening_result']['confidence']}")
    print("Recommended interventions:")
    for intervention in result['intervention_recommendation']['recommended_interventions']:
        print(f"- {intervention}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

### JavaScript Example

```javascript
async function screenForDyscalculia() {
  const assessmentData = {
    number_recognition: 3,
    number_comparison: 2,
    counting_skills: 4,
    place_value: 2,
    calculation_accuracy: 2,
    calculation_fluency: 1,
    arithmetic_facts_recall: 2,
    word_problem_solving: 1,
    working_memory_score: "low",
    visual_spatial_score: "normal",
    math_anxiety_level: "high",
    attention_score: "normal"
  };

  try {
    const response = await fetch('http://localhost:5000/api/screen', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(assessmentData)
    });

    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }

    const result = await response.json();
    
    console.log(`Screening result: ${result.screening_result.prediction}`);
    console.log(`Confidence: ${result.screening_result.confidence}`);
    console.log('Recommended interventions:');
    result.intervention_recommendation.recommended_interventions.forEach(
      intervention => console.log(`- ${intervention}`)
    );
    
    return result;
  } catch (error) {
    console.error('Error:', error);
  }
}

screenForDyscalculia();
```

## Future API Enhancements

The following enhancements are planned for future API versions:

1. OAuth 2.0 authentication
2. WebSocket support for real-time updates
3. Batch processing endpoints
4. Enhanced error reporting and diagnostics
5. Comprehensive metric tracking
6. Multi-language support
7. Enhanced visualization endpoints