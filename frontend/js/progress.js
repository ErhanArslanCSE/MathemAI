/**
 * File: frontend/js/progress.js
 * Progress tracking functionality for MathemAI
 */

// Store active student data
let activeStudentData = null;
let progressChartInstance = null;
let interventionChartInstance = null;

/**
 * Initialize the progress tracking page
 */
function initializeProgress() {
    // Check if progress container exists
    const progressContainer = document.getElementById('progress-container');
    if (!progressContainer) return;
    
    // Load sample student data if not authenticated
    if (!isAuthenticated()) {
        loadSampleData();
        return;
    }
    
    // Try to get student ID from URL
    const urlParams = new URLSearchParams(window.location.search);
    const studentId = urlParams.get('student_id');
    
    if (studentId) {
        loadStudentData(studentId);
    } else {
        loadStudentList();
    }
    
    // Set up event listeners
    setupEventListeners();
}

/**
 * Load sample data for demonstration
 */
function loadSampleData() {
    const progressContainer = document.getElementById('progress-container');
    if (!progressContainer) return;
    
    // Create sample student data
    const sampleStudent = {
        id: 'sample',
        name: 'Sample Student',
        age: 8,
        grade: 3,
        diagnosis: 'math_difficulty',
        interventions: [
            {
                id: 'INT001',
                type: 'visual_aids',
                startDate: '2025-01-15',
                endDate: '2025-02-26',
                preScore: 45,
                postScore: 58
            },
            {
                id: 'INT002',
                type: 'game_based_learning',
                startDate: '2025-03-10',
                endDate: '2025-04-21',
                preScore: 58,
                postScore: 67
            }
        ],
        assessments: [
            {
                date: '2025-01-10',
                scores: {
                    number_recognition: 3,
                    number_comparison: 4,
                    counting_skills: 4,
                    calculation_accuracy: 2,
                    calculation_fluency: 1,
                    arithmetic_facts_recall: 2,
                    word_problem_solving: 1,
                    place_value: 3
                }
            },
            {
                date: '2025-03-01',
                scores: {
                    number_recognition: 4,
                    number_comparison: 4,
                    counting_skills: 5,
                    calculation_accuracy: 3,
                    calculation_fluency: 2,
                    arithmetic_facts_recall: 3,
                    word_problem_solving: 2,
                    place_value: 3
                }
            },
            {
                date: '2025-04-25',
                scores: {
                    number_recognition: 4,
                    number_comparison: 5,
                    counting_skills: 5,
                    calculation_accuracy: 4,
                    calculation_fluency: 3,
                    arithmetic_facts_recall: 3,
                    word_problem_solving: 3,
                    place_value: 4
                }
            }
        ]
    };
    
    // Set active student data
    activeStudentData = sampleStudent;
    
    // Display student data
    displayStudentProgress(sampleStudent);
}

/**
 * Load list of students
 */
async function loadStudentList() {
    const progressContainer = document.getElementById('progress-container');
    if (!progressContainer) return;
    
    // Show loading indicator
    progressContainer.innerHTML = `
        <div class="text-center">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p>Loading student list...</p>
        </div>
    `;
    
    try {
        // In a real application, you would fetch this from the API
        // For demonstration, we'll use a fixed list
        const students = [
            { id: 1, name: 'Emma Johnson', grade: 3, diagnosis: 'dyscalculia' },
            { id: 2, name: 'Liam Williams', grade: 4, diagnosis: 'math_difficulty' },
            { id: 3, name: 'Olivia Smith', grade: 2, diagnosis: 'dyscalculia' },
            { id: 4, name: 'Noah Brown', grade: 5, diagnosis: 'typical' },
            { id: 5, name: 'Ava Davis', grade: 3, diagnosis: 'math_difficulty' }
        ];
        
        displayStudentList(students);
    } catch (error) {
        console.error('Error loading student list:', error);
        progressContainer.innerHTML = `
            <div class="alert alert-danger" role="alert">
                <h4 class="alert-heading">Error Loading Student List</h4>
                <p>There was a problem loading the student list. Please try again later.</p>
                <hr>
                <p class="mb-0">Error details: ${error.message}</p>
                <button class="btn btn-primary mt-3" onclick="loadStudentList()">Try Again</button>
            </div>
        `;
    }
}

/**
 * Display list of students
 * @param {Array} students - List of student objects
 */
function displayStudentList(students) {
    const progressContainer = document.getElementById('progress-container');
    if (!progressContainer) return;
    
    let html = `
        <h2 class="mb-4">Student Progress Tracking</h2>
        <div class="card mb-4">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h3 class="mb-0">Students</h3>
				<button class="btn btn-primary btn-sm" id="add-student-btn">
                    <i class="fas fa-plus"></i> Add Student
                </button>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-hover">
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>Grade</th>
                                <th>Diagnosis</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
    `;
    
    // Add rows for each student
    students.forEach(student => {
        let diagnosisClass = '';
        let diagnosisText = '';
        
        switch (student.diagnosis) {
            case 'dyscalculia':
                diagnosisClass = 'badge bg-danger';
                diagnosisText = 'Dyscalculia';
                break;
            case 'math_difficulty':
                diagnosisClass = 'badge bg-warning text-dark';
                diagnosisText = 'Math Difficulty';
                break;
            case 'typical':
                diagnosisClass = 'badge bg-success';
                diagnosisText = 'Typical';
                break;
            default:
                diagnosisClass = 'badge bg-secondary';
                diagnosisText = 'Not Assessed';
        }
        
        html += `
            <tr>
                <td>${student.name}</td>
                <td>${student.grade}</td>
                <td><span class="${diagnosisClass}">${diagnosisText}</span></td>
                <td>
                    <a href="?student_id=${student.id}" class="btn btn-sm btn-primary">
                        <i class="fas fa-chart-line"></i> View Progress
                    </a>
                </td>
            </tr>
        `;
    });
    
    html += `
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <div class="alert alert-info" role="alert">
            <i class="fas fa-info-circle"></i> Select a student to view their detailed progress.
        </div>
    `;
    
    progressContainer.innerHTML = html;
    
    // Add event listener to add student button
    const addStudentBtn = document.getElementById('add-student-btn');
    if (addStudentBtn) {
        addStudentBtn.addEventListener('click', showAddStudentForm);
    }
}

/**
 * Show form to add a new student
 */
function showAddStudentForm() {
    // Create modal if it doesn't exist
    let modal = document.getElementById('add-student-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'add-student-modal';
        modal.className = 'modal fade';
        modal.setAttribute('tabindex', '-1');
        modal.setAttribute('aria-labelledby', 'add-student-modal-label');
        modal.setAttribute('aria-hidden', 'true');
        
        modal.innerHTML = `
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="add-student-modal-label">Add New Student</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <form id="add-student-form">
                            <div class="mb-3">
                                <label for="student-name" class="form-label">Student Name</label>
                                <input type="text" class="form-control" id="student-name" required>
                            </div>
                            <div class="mb-3">
                                <label for="student-age" class="form-label">Age</label>
                                <input type="number" class="form-control" id="student-age" min="5" max="18" required>
                            </div>
                            <div class="mb-3">
                                <label for="student-grade" class="form-label">Grade</label>
                                <select class="form-select" id="student-grade" required>
                                    <option value="">Select Grade</option>
                                    <option value="1">1st Grade</option>
                                    <option value="2">2nd Grade</option>
                                    <option value="3">3rd Grade</option>
                                    <option value="4">4th Grade</option>
                                    <option value="5">5th Grade</option>
                                    <option value="6">6th Grade</option>
                                </select>
                            </div>
                        </form>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="button" class="btn btn-primary" id="save-student-btn">Add Student</button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Add event listener to save button
        const saveButton = document.getElementById('save-student-btn');
        if (saveButton) {
            saveButton.addEventListener('click', saveNewStudent);
        }
    }
    
    // Show the modal
    const modalInstance = new bootstrap.Modal(modal);
    modalInstance.show();
}

/**
 * Save a new student
 */
function saveNewStudent() {
    const form = document.getElementById('add-student-form');
    if (!form) return;
    
    // Check form validity
    if (!form.checkValidity()) {
        form.reportValidity();
        return;
    }
    
    // Get form data
    const name = document.getElementById('student-name').value;
    const age = document.getElementById('student-age').value;
    const grade = document.getElementById('student-grade').value;
    
    // In a real application, you would save this to the API
    // For now, we'll just show a notification
    showNotification(`Added student: ${name}`, 'success');
    
    // Close the modal
    const modal = bootstrap.Modal.getInstance(document.getElementById('add-student-modal'));
    if (modal) {
        modal.hide();
    }
    
    // Reload student list (in a real app, you'd add the new student to the list)
    loadStudentList();
}

/**
 * Load data for a specific student
 * @param {string|number} studentId - ID of the student to load
 */
async function loadStudentData(studentId) {
    const progressContainer = document.getElementById('progress-container');
    if (!progressContainer) return;
    
    // Show loading indicator
    progressContainer.innerHTML = `
        <div class="text-center">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p>Loading student data...</p>
        </div>
    `;
    
    try {
        // In a real application, you would fetch this from the API
        // For demonstration, we'll create sample data
        const sampleStudent = {
            id: studentId,
            name: `Student ${studentId}`,
            age: 8 + (studentId % 3),
            grade: 3 + (studentId % 3),
            diagnosis: ['dyscalculia', 'math_difficulty', 'typical'][studentId % 3],
            interventions: [
                {
                    id: 'INT001',
                    type: 'visual_aids',
                    startDate: '2025-01-15',
                    endDate: '2025-02-26',
                    preScore: 45,
                    postScore: 58
                },
                {
                    id: 'INT002',
                    type: 'game_based_learning',
                    startDate: '2025-03-10',
                    endDate: '2025-04-21',
                    preScore: 58,
                    postScore: 67
                }
            ],
            assessments: [
                {
                    date: '2025-01-10',
                    scores: {
                        number_recognition: 3,
                        number_comparison: 4,
                        counting_skills: 4,
                        calculation_accuracy: 2,
                        calculation_fluency: 1,
                        arithmetic_facts_recall: 2,
                        word_problem_solving: 1,
                        place_value: 3
                    }
                },
                {
                    date: '2025-03-01',
                    scores: {
                        number_recognition: 4,
                        number_comparison: 4,
                        counting_skills: 5,
                        calculation_accuracy: 3,
                        calculation_fluency: 2,
                        arithmetic_facts_recall: 3,
                        word_problem_solving: 2,
                        place_value: 3
                    }
                },
                {
                    date: '2025-04-25',
                    scores: {
                        number_recognition: 4,
                        number_comparison: 5,
                        counting_skills: 5,
                        calculation_accuracy: 4,
                        calculation_fluency: 3,
                        arithmetic_facts_recall: 3,
                        word_problem_solving: 3,
                        place_value: 4
                    }
                }
            ]
        };
        
        // Set active student data
        activeStudentData = sampleStudent;
        
        // Display student data
        displayStudentProgress(sampleStudent);
    } catch (error) {
        console.error('Error loading student data:', error);
        progressContainer.innerHTML = `
            <div class="alert alert-danger" role="alert">
                <h4 class="alert-heading">Error Loading Student Data</h4>
                <p>There was a problem loading the student data. Please try again later.</p>
                <hr>
                <p class="mb-0">Error details: ${error.message}</p>
                <div class="mt-3">
                    <button class="btn btn-primary me-2" onclick="loadStudentData('${studentId}')">Try Again</button>
                    <a href="progress.html" class="btn btn-secondary">Back to Student List</a>
                </div>
            </div>
        `;
    }
}

/**
 * Display progress data for a specific student
 * @param {Object} student - Student data object
 */
function displayStudentProgress(student) {
    const progressContainer = document.getElementById('progress-container');
    if (!progressContainer) return;
    
    let diagnosisClass = '';
    let diagnosisText = '';
    
    switch (student.diagnosis) {
        case 'dyscalculia':
            diagnosisClass = 'badge bg-danger';
            diagnosisText = 'Dyscalculia';
            break;
        case 'math_difficulty':
            diagnosisClass = 'badge bg-warning text-dark';
            diagnosisText = 'Math Difficulty';
            break;
        case 'typical':
            diagnosisClass = 'badge bg-success';
            diagnosisText = 'Typical';
            break;
        default:
            diagnosisClass = 'badge bg-secondary';
            diagnosisText = 'Not Assessed';
    }
    
    const html = `
        <div class="d-flex justify-content-between align-items-center mb-4">
            <h2>Student Progress</h2>
            <a href="progress.html" class="btn btn-outline-secondary">
                <i class="fas fa-arrow-left"></i> Back to Student List
            </a>
        </div>
        
        <div class="card mb-4">
            <div class="card-header">
                <h3>Student Information</h3>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <p><strong>Name:</strong> ${student.name}</p>
                        <p><strong>Age:</strong> ${student.age}</p>
                        <p><strong>Grade:</strong> ${student.grade}</p>
                    </div>
                    <div class="col-md-6">
                        <p><strong>Diagnosis:</strong> <span class="${diagnosisClass}">${diagnosisText}</span></p>
                        <p><strong>Assessments:</strong> ${student.assessments.length}</p>
                        <p><strong>Interventions:</strong> ${student.interventions.length}</p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mb-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h3>Skills Progress Over Time</h3>
                        <div class="form-check form-switch">
                            <input class="form-check-input" type="checkbox" id="show-all-skills" checked>
                            <label class="form-check-label" for="show-all-skills">Show All Skills</label>
                        </div>
                    </div>
                    <div class="card-body">
                        <canvas id="progress-chart"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mb-4">
            <div class="col-md-6">
                <div class="card h-100">
                    <div class="card-header">
                        <h3>Intervention Effectiveness</h3>
                    </div>
                    <div class="card-body">
                        <canvas id="intervention-chart"></canvas>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card h-100">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h3>Interventions</h3>
                        <button class="btn btn-primary btn-sm" id="add-intervention-btn">
                            <i class="fas fa-plus"></i> Add
                        </button>
                    </div>
                    <div class="card-body">
                        <div id="interventions-list"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card mb-4">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h3>Assessment History</h3>
                <button class="btn btn-primary btn-sm" id="add-assessment-btn">
                    <i class="fas fa-plus"></i> Add Assessment
                </button>
            </div>
            <div class="card-body">
                <div id="assessment-history"></div>
            </div>
        </div>
    `;
    
    progressContainer.innerHTML = html;
    
    // Display interventions
    displayInterventions(student.interventions);
    
    // Display assessment history
    displayAssessmentHistory(student.assessments);
    
    // Create charts
    createProgressChart(student.assessments);
    createInterventionChart(student.interventions);
    
    // Set up event listeners for toggle switches
    setupSkillToggle();
    
    // Add event listeners for buttons
    const addInterventionBtn = document.getElementById('add-intervention-btn');
    if (addInterventionBtn) {
        addInterventionBtn.addEventListener('click', showAddInterventionForm);
    }
    
    const addAssessmentBtn = document.getElementById('add-assessment-btn');
    if (addAssessmentBtn) {
        addAssessmentBtn.addEventListener('click', showAddAssessmentForm);
    }
}

/**
 * Display intervention list
 * @param {Array} interventions - List of intervention objects
 */
function displayInterventions(interventions) {
    const interventionsContainer = document.getElementById('interventions-list');
    if (!interventionsContainer) return;
    
    if (interventions.length === 0) {
        interventionsContainer.innerHTML = `
            <div class="alert alert-info">
                No interventions recorded yet.
            </div>
        `;
        return;
    }
    
    let html = `
        <div class="list-group">
    `;
    
    interventions.forEach(intervention => {
        const improvementPercent = Math.round(((intervention.postScore - intervention.preScore) / intervention.preScore) * 100);
        const improvementClass = improvementPercent >= 10 ? 'text-success' : (improvementPercent >= 5 ? 'text-warning' : 'text-danger');
        
        // Format the intervention type
        const interventionType = intervention.type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        
        html += `
            <div class="list-group-item">
                <div class="d-flex w-100 justify-content-between">
                    <h5 class="mb-1">${interventionType}</h5>
                    <span class="badge bg-primary">${intervention.id}</span>
                </div>
                <p class="mb-1">
                    <small>
                        ${formatDate(intervention.startDate)} - ${formatDate(intervention.endDate)}
                    </small>
                </p>
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <small>Pre-score: ${intervention.preScore}</small> →
                        <small>Post-score: ${intervention.postScore}</small>
                    </div>
                    <span class="${improvementClass}">
                        <i class="fas fa-arrow-${improvementPercent >= 0 ? 'up' : 'down'}"></i>
                        ${Math.abs(improvementPercent)}%
                    </span>
                </div>
            </div>
        `;
    });
    
    html += `
        </div>
    `;
    
    interventionsContainer.innerHTML = html;
}

/**
 * Display assessment history
 * @param {Array} assessments - List of assessment objects
 */
function displayAssessmentHistory(assessments) {
    const historyContainer = document.getElementById('assessment-history');
    if (!historyContainer) return;
    
    if (assessments.length === 0) {
        historyContainer.innerHTML = `
            <div class="alert alert-info">
                No assessments recorded yet.
            </div>
        `;
        return;
    }
    
    let html = `
        <div class="table-responsive">
            <table class="table table-hover">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Number Skills</th>
                        <th>Calculation Skills</th>
                        <th>Problem Solving</th>
                        <th>Overall</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
    `;
    
    // Sort assessments by date (most recent first)
    const sortedAssessments = [...assessments].sort((a, b) => new Date(b.date) - new Date(a.date));
    
    sortedAssessments.forEach(assessment => {
        const scores = assessment.scores;
        
        // Calculate averages for skill groups
        const numberSkillsAvg = ((scores.number_recognition + scores.number_comparison + scores.counting_skills) / 3).toFixed(1);
        const calculationSkillsAvg = ((scores.calculation_accuracy + scores.calculation_fluency + scores.arithmetic_facts_recall) / 3).toFixed(1);
        const problemSolvingAvg = ((scores.word_problem_solving + scores.place_value) / 2).toFixed(1);
        
        // Calculate overall average
        const overallAvg = (
            (scores.number_recognition + scores.number_comparison + scores.counting_skills +
             scores.calculation_accuracy + scores.calculation_fluency + scores.arithmetic_facts_recall +
             scores.word_problem_solving + scores.place_value) / 8
        ).toFixed(1);
        
        html += `
            <tr>
                <td>${formatDate(assessment.date)}</td>
                <td>${numberSkillsAvg}/5</td>
                <td>${calculationSkillsAvg}/5</td>
                <td>${problemSolvingAvg}/5</td>
                <td>${overallAvg}/5</td>
                <td>
                    <button class="btn btn-sm btn-outline-primary view-assessment-btn" data-date="${assessment.date}">
                        <i class="fas fa-eye"></i> View
                    </button>
                </td>
            </tr>
        `;
    });
    
    html += `
                </tbody>
            </table>
        </div>
    `;
    
    historyContainer.innerHTML = html;
    
    // Add event listeners to view buttons
    const viewButtons = document.querySelectorAll('.view-assessment-btn');
    viewButtons.forEach(button => {
        button.addEventListener('click', () => {
            const date = button.getAttribute('data-date');
            showAssessmentDetails(date);
        });
    });
}

/**
 * Show detailed information for a specific assessment
 * @param {string} date - Date of the assessment to show
 */
function showAssessmentDetails(date) {
    if (!activeStudentData) return;
    
    // Find the assessment by date
    const assessment = activeStudentData.assessments.find(a => a.date === date);
    if (!assessment) return;
    
    // Create modal if it doesn't exist
    let modal = document.getElementById('assessment-details-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'assessment-details-modal';
        modal.className = 'modal fade';
        modal.setAttribute('tabindex', '-1');
        modal.setAttribute('aria-labelledby', 'assessment-details-modal-label');
        modal.setAttribute('aria-hidden', 'true');
        
        modal.innerHTML = `
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="assessment-details-modal-label">Assessment Details</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body" id="assessment-details-content">
                        <!-- Content will be inserted here -->
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
    }
    
    // Update modal content
    const modalContent = document.getElementById('assessment-details-content');
    if (modalContent) {
        modalContent.innerHTML = `
            <h4>Assessment on ${formatDate(assessment.date)}</h4>
            
            <table class="table table-bordered mt-3">
                <thead>
                    <tr>
                        <th>Skill</th>
                        <th>Score</th>
                        <th>Interpretation</th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="table-primary">
                        <th colspan="3">Number Skills</th>
                    </tr>
                    ${createSkillRow('Number Recognition', assessment.scores.number_recognition)}
                    ${createSkillRow('Number Comparison', assessment.scores.number_comparison)}
                    ${createSkillRow('Counting Skills', assessment.scores.counting_skills)}
                    
                    <tr class="table-primary">
                        <th colspan="3">Calculation Skills</th>
                    </tr>
                    ${createSkillRow('Calculation Accuracy', assessment.scores.calculation_accuracy)}
                    ${createSkillRow('Calculation Fluency', assessment.scores.calculation_fluency)}
                    ${createSkillRow('Arithmetic Facts Recall', assessment.scores.arithmetic_facts_recall)}
                    
                    <tr class="table-primary">
                        <th colspan="3">Problem Solving</th>
                    </tr>
                    ${createSkillRow('Word Problem Solving', assessment.scores.word_problem_solving)}
                    ${createSkillRow('Place Value Understanding', assessment.scores.place_value)}
                </tbody>
            </table>
        `;
    }
    
    // Show the modal
    const modalInstance = new bootstrap.Modal(modal);
    modalInstance.show();
}

/**
 * Create a table row for skill information
 * @param {string} skillName - Name of the skill
 * @param {number} score - Score (1-5)
 * @return {string} HTML for the table row
 */
function createSkillRow(skillName, score) {
    let interpretation = '';
    let rowClass = '';
    
    // Determine interpretation and class based on score
    if (score <= 2) {
        interpretation = 'Significant Challenge - May require intensive intervention';
        rowClass = 'table-danger';
    } else if (score === 3) {
        interpretation = 'Some Difficulty - Could benefit from targeted support';
        rowClass = 'table-warning';
    } else if (score === 4) {
        interpretation = 'Adequate - Within typical range';
        rowClass = 'table-light';
    } else {
        interpretation = 'Strength - Above average performance';
        rowClass = 'table-success';
    }
    
    return `
        <tr class="${rowClass}">
            <td>${skillName}</td>
            <td>${score}/5</td>
            <td>${interpretation}</td>
        </tr>
    `;
}

/**
 * Create a chart showing progress over time
 * @param {Array} assessments - List of assessment objects
 */
function createProgressChart(assessments) {
    const ctx = document.getElementById('progress-chart');
    if (!ctx) return;
    
    // Sort assessments by date
    const sortedAssessments = [...assessments].sort((a, b) => new Date(a.date) - new Date(b.date));
    
    // Prepare data for chart
    const labels = sortedAssessments.map(a => formatDate(a.date));
    
    // Define skill categories and their colors
    const skillsConfig = {
        number_recognition: { label: 'Number Recognition', color: 'rgba(75, 192, 192, 1)' },
        number_comparison: { label: 'Number Comparison', color: 'rgba(54, 162, 235, 1)' },
        counting_skills: { label: 'Counting Skills', color: 'rgba(153, 102, 255, 1)' },
        calculation_accuracy: { label: 'Calculation Accuracy', color: 'rgba(255, 99, 132, 1)' },
        calculation_fluency: { label: 'Calculation Fluency', color: 'rgba(255, 159, 64, 1)' },
        arithmetic_facts_recall: { label: 'Arithmetic Facts', color: 'rgba(255, 205, 86, 1)' },
        word_problem_solving: { label: 'Problem Solving', color: 'rgba(201, 203, 207, 1)' },
        place_value: { label: 'Place Value', color: 'rgba(100, 120, 140, 1)' }
    };
    
    // Create datasets
    const datasets = Object.entries(skillsConfig).map(([key, config]) => {
        return {
            label: config.label,
            data: sortedAssessments.map(a => a.scores[key]),
            borderColor: config.color,
            backgroundColor: config.color.replace('1)', '0.2)'),
            tension: 0.1
        };
    });
    
    // Destroy previous chart if it exists
    if (progressChartInstance) {
        progressChartInstance.destroy();
    }
    
    // Create chart
    progressChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 5,
                    title: {
                        display: true,
                        text: 'Score (1-5)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Assessment Date'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Skill Progress Over Time'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            }
        }
    });
}

/**
 * Create a chart showing intervention effectiveness
 * @param {Array} interventions - List of intervention objects
 */
function createInterventionChart(interventions) {
    const ctx = document.getElementById('intervention-chart');
    if (!ctx) return;
    
    // Sort interventions by start date
    const sortedInterventions = [...interventions].sort((a, b) => new Date(a.startDate) - new Date(b.startDate));
    
    // Prepare data for chart
    const labels = sortedInterventions.map(i => i.type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()));
    const preScores = sortedInterventions.map(i => i.preScore);
    const postScores = sortedInterventions.map(i => i.postScore);
    
    // Destroy previous chart if it exists
    if (interventionChartInstance) {
        interventionChartInstance.destroy();
    }
    
    // Create chart
    interventionChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Pre-Intervention Score',
                    data: preScores,
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Post-Intervention Score',
                    data: postScores,
                    backgroundColor: 'rgba(75, 192, 192, 0.5)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Score (0-100)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Intervention Type'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Intervention Effectiveness'
                }
            }
        }
    });
}

/**
 * Set up skill toggle functionality
 */
function setupSkillToggle() {
    const toggle = document.getElementById('show-all-skills');
    if (!toggle || !progressChartInstance) return;
    
    toggle.addEventListener('change', function() {
        if (this.checked) {
            // Show all datasets
            progressChartInstance.data.datasets.forEach(dataset => {
                dataset.hidden = false;
            });
        } else {
            // Only show number recognition, calculation accuracy, and problem solving
            progressChartInstance.data.datasets.forEach(dataset => {
                dataset.hidden = !['Number Recognition', 'Calculation Accuracy', 'Problem Solving'].includes(dataset.label);
            });
        }
        
        progressChartInstance.update();
    });
}

/**
 * Set up event listeners
 */
function setupEventListeners() {
    // Add event listeners as needed
}

/**
 * Show form to add a new intervention
 */
function showAddInterventionForm() {
    // Create modal content and functionality
    // Implementation depends on application requirements
    showNotification('Add intervention functionality will be implemented in a future update.', 'info');
}

/**
 * Show form to add a new assessment
 */
function showAddAssessmentForm() {
    // Create modal content and functionality
    // Implementation depends on application requirements
    showNotification('Add assessment functionality will be implemented in a future update.', 'info');
}

// Initialize the progress tracking page when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    initializeProgress();
});