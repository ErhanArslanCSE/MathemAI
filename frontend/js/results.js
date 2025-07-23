/**
 * File: frontend/js/results.js
 * Results visualization and management for MathemAI
 */

/**
 * Initializes the results page
 */
function initializeResults() {
    // Retrieve results from localStorage
    const resultsData = localStorage.getItem('mathemat_assessment_results');
    const assessmentData = localStorage.getItem('mathemat_assessment_data');
    
    // Check if results exist
    if (!resultsData || !assessmentData) {
        showNoResultsMessage();
        return;
    }
    
    try {
        // Parse the stored data
        const results = JSON.parse(resultsData);
        const assessment = JSON.parse(assessmentData);
        
        // Display the results
        displayResults(results, assessment);
        
        // Create charts
        createResultsCharts(results, assessment);
        
        // Set up print functionality
        setupPrintButton();
        
        // Set up save functionality
        setupSaveButton(results, assessment);
    } catch (error) {
        console.error('Error displaying results:', error);
        showErrorMessage(error);
    }
}

/**
 * Display the screening results
 * @param {Object} results - The screening results
 * @param {Object} assessment - The assessment data
 */
function displayResults(results, assessment) {
    const resultsContainer = document.getElementById('results-container');
    if (!resultsContainer) return;
    
    // Extract data
    const { screening_result, intervention_recommendation } = results;
    const diagnosis = screening_result.prediction;
    const confidence = screening_result.confidence * 100; // Convert to percentage
    
    // Create heading with diagnosis
    let diagnosisClass = '';
    let diagnosisText = '';
    
    switch (diagnosis) {
        case 'dyscalculia':
            diagnosisClass = 'alert-danger';
            diagnosisText = 'Indicators of Dyscalculia Detected';
            break;
        case 'math_difficulty':
            diagnosisClass = 'alert-warning';
            diagnosisText = 'Mathematical Learning Difficulties Detected';
            break;
        case 'typical':
            diagnosisClass = 'alert-success';
            diagnosisText = 'Typical Mathematical Development';
            break;
        default:
            diagnosisClass = 'alert-info';
            diagnosisText = 'Assessment Results';
    }
    
    // Create results HTML
    resultsContainer.innerHTML = `
        <div class="alert ${diagnosisClass} mb-4" role="alert">
            <h2 class="alert-heading">${diagnosisText}</h2>
            <p>Confidence: ${confidence.toFixed(1)}%</p>
        </div>
        
        <div class="row mb-5">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h3>Strengths and Challenges</h3>
                    </div>
                    <div class="card-body">
                        <div id="skills-chart-container">
                            <canvas id="skills-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h3>Diagnosis Probability</h3>
                    </div>
                    <div class="card-body">
                        <div id="diagnosis-chart-container">
                            <canvas id="diagnosis-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card mb-4">
            <div class="card-header">
                <h3>Recommended Interventions</h3>
            </div>
            <div class="card-body">
                <div id="interventions-container"></div>
            </div>
        </div>
        
        <div class="card mb-4">
            <div class="card-header">
                <h3>Detailed Assessment</h3>
            </div>
            <div class="card-body">
                <div id="detailed-assessment-container"></div>
            </div>
        </div>
        
        <div class="d-flex justify-content-between mt-4">
            <button id="print-results" class="btn btn-outline-secondary">
                <i class="fas fa-print"></i> Print Results
            </button>
            <button id="save-results" class="btn btn-primary">
                <i class="fas fa-save"></i> Save Results
            </button>
        </div>
    `;
    
    // Display recommended interventions
    displayInterventions(intervention_recommendation);
    
    // Display detailed assessment
    displayDetailedAssessment(assessment);
}

/**
 * Display the recommended interventions
 * @param {Object} recommendation - The intervention recommendation
 */
function displayInterventions(recommendation) {
    const interventionsContainer = document.getElementById('interventions-container');
    if (!interventionsContainer || !recommendation) return;
    
    const { recommended_interventions, description } = recommendation;
    
    // Create HTML for interventions
    let interventionsHtml = `
        <p class="mb-4">${description}</p>
        <h4>Recommended Approaches:</h4>
        <div class="row">
    `;
    
    // Create cards for each intervention
    recommended_interventions.forEach(intervention => {
        const title = intervention.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        let iconClass = '';
        let description = '';
        
        // Set icon and description based on intervention type
        switch (intervention) {
            case 'multisensory_approach':
                iconClass = 'fa-hands';
                description = 'Uses tactile and visual methods to help understand mathematical concepts through multiple senses.';
                break;
            case 'visual_aids':
                iconClass = 'fa-images';
                description = 'Uses visual tools like number lines and manipulatives to visualize mathematical relationships.';
                break;
            case 'game_based_learning':
                iconClass = 'fa-gamepad';
                description = 'Incorporates games and play-based activities to make math practice engaging and reduce anxiety.';
                break;
            case 'structured_sequence':
                iconClass = 'fa-list-ol';
                description = 'Provides a highly structured, sequential approach with clear, consistent progression of skills.';
                break;
            case 'technology_assisted':
                iconClass = 'fa-laptop';
                description = 'Uses digital tools and adaptive technology to provide interactive practice with immediate feedback.';
                break;
            default:
                iconClass = 'fa-star';
                description = 'Personalized approach based on individual learning profile.';
        }
        
        // Add intervention card
        interventionsHtml += `
            <div class="col-md-4 mb-3">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <div class="intervention-icon mb-3">
                            <i class="fas ${iconClass} fa-2x"></i>
                        </div>
                        <h5 class="card-title">${title}</h5>
                        <p class="card-text">${description}</p>
                    </div>
                </div>
            </div>
        `;
    });
    
    interventionsHtml += `</div>`;
    
    // Set the HTML
    interventionsContainer.innerHTML = interventionsHtml;
}

/**
 * Display detailed assessment information
 * @param {Object} assessment - The assessment data
 */
function displayDetailedAssessment(assessment) {
    const detailedContainer = document.getElementById('detailed-assessment-container');
    if (!detailedContainer) return;
    
    // Create HTML table for detailed assessment
    let assessmentHtml = `
        <table class="table table-bordered">
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Skill</th>
                    <th>Score</th>
                    <th>Interpretation</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    // Number skills
    assessmentHtml += createAssessmentRow(
        'Number Skills', 
        'Number Recognition', 
        assessment.number_recognition, 
        getInterpretation(assessment.number_recognition)
    );
    
    assessmentHtml += createAssessmentRow(
        'Number Skills', 
        'Number Comparison', 
        assessment.number_comparison, 
        getInterpretation(assessment.number_comparison)
    );
    
    assessmentHtml += createAssessmentRow(
        'Number Skills', 
        'Counting Skills', 
        assessment.counting_skills, 
        getInterpretation(assessment.counting_skills)
    );
    
    // Calculation skills
    assessmentHtml += createAssessmentRow(
        'Calculation', 
        'Calculation Accuracy', 
        assessment.calculation_accuracy, 
        getInterpretation(assessment.calculation_accuracy)
    );
    
    assessmentHtml += createAssessmentRow(
        'Calculation', 
        'Calculation Fluency', 
        assessment.calculation_fluency, 
        getInterpretation(assessment.calculation_fluency)
    );
    
    assessmentHtml += createAssessmentRow(
        'Calculation', 
        'Arithmetic Facts Recall', 
        assessment.arithmetic_facts_recall, 
        getInterpretation(assessment.arithmetic_facts_recall)
    );
    
    // Problem solving
    assessmentHtml += createAssessmentRow(
        'Problem Solving', 
        'Word Problem Solving', 
        assessment.word_problem_solving, 
        getInterpretation(assessment.word_problem_solving)
    );
    
    assessmentHtml += createAssessmentRow(
        'Problem Solving', 
        'Place Value Understanding', 
        assessment.place_value, 
        getInterpretation(assessment.place_value)
    );
    
    // Cognitive factors
    assessmentHtml += createAssessmentRow(
        'Cognitive Factors', 
        'Math Anxiety Level', 
        assessment.math_anxiety_level, 
        getAnxietyInterpretation(assessment.math_anxiety_level)
    );
    
    assessmentHtml += createAssessmentRow(
        'Cognitive Factors', 
        'Working Memory', 
        assessment.working_memory_score, 
        getCognitiveInterpretation(assessment.working_memory_score)
    );
    
    assessmentHtml += createAssessmentRow(
        'Cognitive Factors', 
        'Visual-Spatial Skills', 
        assessment.visual_spatial_score, 
        getCognitiveInterpretation(assessment.visual_spatial_score)
    );
    
    assessmentHtml += createAssessmentRow(
        'Cognitive Factors', 
        'Attention', 
        assessment.attention_score, 
        getCognitiveInterpretation(assessment.attention_score)
    );
    
    assessmentHtml += `
            </tbody>
        </table>
    `;
    
    // Set the HTML
    detailedContainer.innerHTML = assessmentHtml;
}

/**
 * Create a table row for assessment data
 * @param {string} category - The category of the skill
 * @param {string} skill - The name of the skill
 * @param {*} value - The score or value
 * @param {string} interpretation - The interpretation of the score
 * @return {string} HTML for the table row
 */
function createAssessmentRow(category, skill, value, interpretation) {
    let displayValue = value;
    let rowClass = '';
    
    // Determine row class based on interpretation
    if (interpretation.includes('Significant Challenge')) {
        rowClass = 'table-danger';
    } else if (interpretation.includes('Some Difficulty')) {
        rowClass = 'table-warning';
    } else if (interpretation.includes('Strength')) {
        rowClass = 'table-success';
    }
    
    return `
        <tr class="${rowClass}">
            <td>${category}</td>
            <td>${skill}</td>
            <td>${displayValue}</td>
            <td>${interpretation}</td>
        </tr>
    `;
}

/**
 * Get interpretation of numeric scores
 * @param {number} score - Score on a 1-5 scale
 * @return {string} Interpretation text
 */
function getInterpretation(score) {
    if (score <= 2) {
        return 'Significant Challenge - May require intensive intervention';
    } else if (score === 3) {
        return 'Some Difficulty - Could benefit from targeted support';
    } else if (score === 4) {
        return 'Adequate - Within typical range';
    } else {
        return 'Strength - Above average performance';
    }
}

/**
 * Get interpretation of anxiety levels
 * @param {string} level - Anxiety level
 * @return {string} Interpretation text
 */
function getAnxietyInterpretation(level) {
    switch (level) {
        case 'high':
            return 'High anxiety - May significantly impact performance';
        case 'medium':
            return 'Moderate anxiety - May sometimes impact performance';
        case 'low':
            return 'Low anxiety - Minimal impact on performance';
        default:
            return 'Not assessed';
    }
}

/**
 * Get interpretation of cognitive scores
 * @param {string} score - Cognitive measure
 * @return {string} Interpretation text
 */
function getCognitiveInterpretation(score) {
    switch (score) {
        case 'normal':
            return 'Typical range - No significant concerns';
        case 'low':
            return 'Below average - May impact mathematical learning';
        case 'very_low':
            return 'Significantly below average - Likely impacts mathematical learning';
        default:
            return 'Not assessed';
    }
}

/**
 * Create charts to visualize the results
 * @param {Object} results - The screening results
 * @param {Object} assessment - The assessment data
 */
function createResultsCharts(results, assessment) {
    // Skills chart
    createSkillsChart(assessment);
    
    // Diagnosis probability chart
    createDiagnosisChart(results.screening_result.probabilities);
}

/**
 * Create a chart showing skills assessment
 * @param {Object} assessment - The assessment data
 */
function createSkillsChart(assessment) {
    const ctx = document.getElementById('skills-chart');
    if (!ctx) return;
    
    // Extract skill data
    const skillData = [
        assessment.number_recognition,
        assessment.number_comparison,
        assessment.counting_skills,
        assessment.calculation_accuracy,
        assessment.calculation_fluency,
        assessment.arithmetic_facts_recall,
        assessment.word_problem_solving,
        assessment.place_value
    ];
    
    // Skill labels
    const skillLabels = [
        'Number Recognition',
        'Number Comparison',
        'Counting Skills',
        'Calculation Accuracy',
        'Calculation Fluency',
        'Arithmetic Facts',
        'Problem Solving',
        'Place Value'
    ];
    
    // Create chart
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: skillLabels,
            datasets: [{
                label: 'Skill Level',
                data: skillData,
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgb(54, 162, 235)',
                pointBackgroundColor: 'rgb(54, 162, 235)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgb(54, 162, 235)'
            }]
        },
        options: {
            scales: {
                r: {
                    angleLines: {
                        display: true
                    },
                    suggestedMin: 0,
                    suggestedMax: 5
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Mathematical Skills Profile'
                }
            }
        }
    });
}

/**
 * Create a chart showing diagnosis probabilities
 * @param {Object} probabilities - The diagnosis probabilities
 */
function createDiagnosisChart(probabilities) {
    const ctx = document.getElementById('diagnosis-chart');
    if (!ctx) return;
    
    // Labels and data
    const labels = [];
    const data = [];
    const backgroundColors = [];
    
    // Process probabilities
    for (const [diagnosis, probability] of Object.entries(probabilities)) {
        // Format the label
        const label = diagnosis.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        labels.push(label);
        
        // Convert to percentage
        data.push(probability * 100);
        
        // Set color based on diagnosis
        let color;
        switch (diagnosis) {
            case 'dyscalculia':
                color = 'rgba(220, 53, 69, 0.8)'; // Danger/red
                break;
            case 'math_difficulty':
                color = 'rgba(255, 193, 7, 0.8)'; // Warning/yellow
                break;
            case 'typical':
                color = 'rgba(40, 167, 69, 0.8)'; // Success/green
                break;
            default:
                color = 'rgba(23, 162, 184, 0.8)'; // Info/blue
        }
        backgroundColors.push(color);
    }
    
    // Create chart
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Probability (%)',
                data: data,
                backgroundColor: backgroundColors,
                borderColor: backgroundColors.map(c => c.replace('0.8', '1')),
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Probability (%)'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Diagnosis Probability'
                }
            }
        }
    });
}

/**
 * Set up the print button functionality
 */
function setupPrintButton() {
    const printButton = document.getElementById('print-results');
    if (!printButton) return;
    
    printButton.addEventListener('click', () => {
        window.print();
    });
}

/**
 * Set up the save button functionality
 * @param {Object} results - The screening results
 * @param {Object} assessment - The assessment data
 */
function setupSaveButton(results, assessment) {
    const saveButton = document.getElementById('save-results');
    if (!saveButton) return;
    
    saveButton.addEventListener('click', async () => {
        try {
            // Save the assessment and results to the database
            await api.saveAssessment({
                ...assessment,
                screening_result: results.screening_result.prediction
            });
            
            showNotification('Results saved successfully!', 'success');
        } catch (error) {
            console.error('Error saving results:', error);
            showNotification('Failed to save results. Please try again.', 'error');
        }
    });
}

/**
 * Show a message when no results are available
 */
function showNoResultsMessage() {
    const resultsContainer = document.getElementById('results-container');
    if (!resultsContainer) return;
    
    resultsContainer.innerHTML = `
        <div class="alert alert-info" role="alert">
            <h4 class="alert-heading">No Results Available</h4>
            <p>You haven't completed an assessment yet. Please complete an assessment to see your results.</p>
            <hr>
            <a href="assessment.html" class="btn btn-primary">Take Assessment</a>
        </div>
    `;
}

/**
 * Show an error message
 * @param {Error} error - The error that occurred
 */
function showErrorMessage(error) {
    const resultsContainer = document.getElementById('results-container');
    if (!resultsContainer) return;
    
    resultsContainer.innerHTML = `
        <div class="alert alert-danger" role="alert">
            <h4 class="alert-heading">Error Loading Results</h4>
            <p>There was a problem loading your assessment results. Please try again later.</p>
            <hr>
            <p class="mb-0">Error details: ${error.message}</p>
            <a href="assessment.html" class="btn btn-primary mt-3">Start New Assessment</a>
        </div>
    `;
}

// Initialize the results page when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Check if we're on the results page
    if (document.getElementById('results-container')) {
        initializeResults();
    }
});