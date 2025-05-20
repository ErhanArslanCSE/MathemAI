import pandas as pd
import numpy as np
import os
import datetime
import random
import argparse

def create_directory_if_not_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def save_dataset(df, filename, directory='../datasets'):
    """Save a dataset to CSV file."""
    create_directory_if_not_exists(directory)
    filepath = os.path.join(directory, filename)
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}")
    return filepath

def generate_dyscalculia_assessment_data(n=100, output_file='dyscalculia_assessment_data.csv'):
    """
    Generate a simulated dataset of dyscalculia assessment data.
    
    Parameters:
    -----------
    n : int
        Number of records to generate
    output_file : str
        Name of the output CSV file
        
    Returns:
    --------
    pd.DataFrame
        The generated dataset
    """
    print(f"Generating dyscalculia assessment data with {n} records...")
    
    # Define the distribution of diagnoses
    # 40% dyscalculia, 30% math difficulty, 30% typical
    diagnoses = np.random.choice(
        ['dyscalculia', 'math_difficulty', 'typical'],
        size=n,
        p=[0.4, 0.3, 0.3]
    )
    
    # Create a list to hold the data
    data = []
    
    for i in range(n):
        student_id = i + 1
        diagnosis = diagnoses[i]
        age = random.randint(6, 12)
        grade = max(1, min(7, age - 5))  # Approximate grade based on age
        
        # Set base scores based on diagnosis
        if diagnosis == 'dyscalculia':
            base_range = (1, 4)  # Lower scores
            anxiety_choices = ['high', 'medium']
            anxiety_weights = [0.7, 0.3]
            attention_choices = ['normal', 'low', 'very_low']
            attention_weights = [0.3, 0.5, 0.2]
            memory_choices = ['normal', 'low', 'very_low']
            memory_weights = [0.2, 0.6, 0.2]
            spatial_choices = ['normal', 'low', 'very_low']
            spatial_weights = [0.4, 0.4, 0.2]
            error_patterns = ['transposition', 'reversal', 'miscounting', 
                             'sequence_error', 'operation_confusion', 'multiple_errors']
            response_time = ['slow', 'very_slow']
            response_weights = [0.6, 0.4]
        elif diagnosis == 'math_difficulty':
            base_range = (3, 5)  # Medium scores
            anxiety_choices = ['high', 'medium', 'low']
            anxiety_weights = [0.3, 0.5, 0.2]
            attention_choices = ['normal', 'low', 'very_low']
            attention_weights = [0.5, 0.4, 0.1]
            memory_choices = ['normal', 'low', 'very_low']
            memory_weights = [0.4, 0.5, 0.1]
            spatial_choices = ['normal', 'low']
            spatial_weights = [0.6, 0.4]
            error_patterns = ['calculation_error', 'occasional_error', 'consistent_error', 
                             'operation_confusion']
            response_time = ['average', 'slow']
            response_weights = [0.6, 0.4]
        else:  # typical
            base_range = (4, 5)  # Higher scores
            anxiety_choices = ['medium', 'low']
            anxiety_weights = [0.3, 0.7]
            attention_choices = ['normal', 'low']
            attention_weights = [0.8, 0.2]
            memory_choices = ['normal', 'low']
            memory_weights = [0.9, 0.1]
            spatial_choices = ['normal']
            spatial_weights = [1.0]
            error_patterns = ['rare_error', 'occasional_error', 'none']
            response_time = ['fast', 'average']
            response_weights = [0.7, 0.3]
        
        # Generate assessment scores with some randomness
        number_recognition = random.randint(*base_range)
        number_comparison = random.randint(*base_range)
        counting_skills = random.randint(*base_range)
        place_value = random.randint(*base_range)
        calculation_accuracy = random.randint(*base_range)
        calculation_fluency = max(1, calculation_accuracy - random.randint(0, 2))
        arithmetic_facts_recall = random.randint(*base_range)
        word_problem_solving = max(1, min(5, random.randint(*base_range) - random.randint(0, 2)))
        
        # Generate other attributes
        math_anxiety_level = np.random.choice(anxiety_choices, p=anxiety_weights)
        attention_score = np.random.choice(attention_choices, p=attention_weights)
        working_memory_score = np.random.choice(memory_choices, p=memory_weights)
        visual_spatial_score = np.random.choice(spatial_choices, p=spatial_weights)
        error_pattern = random.choice(error_patterns)
        response_time_val = np.random.choice(response_time, p=response_weights)
        
        data.append({
            'student_id': student_id,
            'age': age,
            'grade': grade,
            'number_recognition': number_recognition,
            'number_comparison': number_comparison,
            'counting_skills': counting_skills,
            'place_value': place_value,
            'calculation_accuracy': calculation_accuracy,
            'calculation_fluency': calculation_fluency,
            'arithmetic_facts_recall': arithmetic_facts_recall,
            'word_problem_solving': word_problem_solving,
            'math_anxiety_level': math_anxiety_level,
            'attention_score': attention_score,
            'working_memory_score': working_memory_score,
            'visual_spatial_score': visual_spatial_score,
            'error_patterns': error_pattern,
            'response_time': response_time_val,
            'diagnosis': diagnosis
        })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    filepath = save_dataset(df, output_file)
    
    print(f"Generated {len(df)} assessment records with the following distribution:")
    print(df['diagnosis'].value_counts())
    
    return df

def generate_error_analysis_data(assessment_df, output_file='error_analysis_data.csv'):
    """
    Generate a simulated dataset of error patterns based on assessment data.
    
    Parameters:
    -----------
    assessment_df : pd.DataFrame
        The assessment data to base errors on
    output_file : str
        Name of the output CSV file
        
    Returns:
    --------
    pd.DataFrame
        The generated dataset
    """
    print("Generating error analysis data...")
    
    # Get students with dyscalculia or math difficulties
    students_with_difficulties = assessment_df[
        assessment_df['diagnosis'].isin(['dyscalculia', 'math_difficulty'])
    ]['student_id'].tolist()
    
    # Define question types and templates
    question_types = {
        'number_recognition': [
            {"question": "Identify the number: {number}", "answer": "{number}"}
        ],
        'number_comparison': [
            {"question": "Which is larger: {num1} or {num2}?", "answer": "{larger}"},
            {"question": "Which is smaller: {num1} or {num2}?", "answer": "{smaller}"}
        ],
        'counting': [
            {"question": "Count from {start} to {end}", "answer": "{sequence}"},
            {"question": "Count by {step}s: {start}, {next}, ?", "answer": "{answer}"}
        ],
        'number_sequence': [
            {"question": "What comes next: {sequence}, ?", "answer": "{next}"}
        ],
        'addition': [
            {"question": "{a} + {b} = ?", "answer": "{sum}"}
        ],
        'subtraction': [
            {"question": "{a} - {b} = ?", "answer": "{difference}"}
        ],
        'multiplication': [
            {"question": "{a} × {b} = ?", "answer": "{product}"}
        ],
        'division': [
            {"question": "{a} ÷ {b} = ?", "answer": "{quotient}"}
        ],
        'place_value': [
            {"question": "What is the ones digit in {number}?", "answer": "{ones}"},
            {"question": "What is the tens digit in {number}?", "answer": "{tens}"}
        ],
        'fractions': [
            {"question": "Which is larger: {frac1} or {frac2}?", "answer": "{larger}"},
            {"question": "What is {fraction} of {whole}?", "answer": "{result}"}
        ],
        'word_problem': [
            {"question": "{name} had {total} {objects}. {action} {change}. How many does {pronoun} have now?", 
             "answer": "{result}"}
        ]
    }
    
    # Create error data
    error_data = []
    question_id_counter = 1
    
    for student_id in students_with_difficulties:
        # Get student details
        student = assessment_df[assessment_df['student_id'] == student_id].iloc[0]
        diagnosis = student['diagnosis']
        error_pattern = student['error_patterns']
        
        # Determine how many questions to generate (more for dyscalculia)
        num_questions = random.randint(4, 6) if diagnosis == 'dyscalculia' else random.randint(3, 5)
        
        # Select question types based on student's profile
        # For example, focus on areas where the student scored lower
        scores = {
            'number_recognition': student['number_recognition'],
            'number_comparison': student['number_comparison'],
            'counting': student['counting_skills'],
            'addition': student['calculation_accuracy'],
            'subtraction': student['calculation_accuracy'],
            'multiplication': student['calculation_fluency'],
            'division': student['calculation_fluency'],
            'place_value': student['place_value'],
            'word_problem': student['word_problem_solving']
        }
        
        # Sort question types by score (ascending) to focus on weaknesses
        sorted_types = sorted(scores.items(), key=lambda x: x[1])
        selected_types = [t[0] for t in sorted_types[:num_questions]]
        
        # Ensure some variety if we have few types
        if len(set(selected_types)) < 3:
            all_types = list(question_types.keys())
            additional_types = random.sample([t for t in all_types if t not in selected_types], 
                                           min(3, len(all_types) - len(set(selected_types))))
            selected_types.extend(additional_types)
            selected_types = selected_types[:num_questions]
        
        # Generate a session date
        session_date = (datetime.datetime(2024, 9, 1) + 
                      datetime.timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d')
        
        # Generate questions and responses
        for q_type in selected_types:
            question_template = random.choice(question_types[q_type])
            question_id = f"Q{question_id_counter}"
            question_id_counter += 1
            
            # Fill in template based on question type
            if q_type == 'number_recognition':
                number = random.randint(10, 99)
                question = question_template['question'].format(number=number)
                correct_answer = str(number)
                
                # For dyscalculia, common error is digit reversal
                if error_pattern in ['transposition', 'reversal'] and random.random() < 0.7:
                    student_answer = str(number)[::-1]  # Reverse the digits
                else:
                    student_answer = correct_answer
                
            elif q_type == 'number_comparison':
                if random.random() < 0.5:
                    num1, num2 = random.randint(5, 90), random.randint(5, 90)
                else:
                    # Create numbers with same digits but different order for confusion
                    tens = random.randint(1, 9)
                    ones = random.randint(0, 9)
                    while tens == ones:
                        ones = random.randint(0, 9)
                    num1 = tens * 10 + ones
                    num2 = ones * 10 + tens
                
                larger = max(num1, num2)
                smaller = min(num1, num2)
                
                if 'larger' in question_template['answer']:
                    question = question_template['question'].format(num1=num1, num2=num2)
                    correct_answer = str(larger)
                    
                    # Common error: choosing the visually "bigger" number
                    if error_pattern in ['operation_confusion', 'reversal'] and random.random() < 0.6:
                        student_answer = str(num1) if num1 != larger else str(num2)
                    else:
                        student_answer = correct_answer
                else:
                    question = question_template['question'].format(num1=num1, num2=num2)
                    correct_answer = str(smaller)
                    
                    # Common error: choosing the visually "smaller" number
                    if error_pattern in ['operation_confusion', 'reversal'] and random.random() < 0.6:
                        student_answer = str(num1) if num1 != smaller else str(num2)
                    else:
                        student_answer = correct_answer
            
            elif q_type == 'counting':
                if 'sequence' in question_template['answer']:
                    start = random.randint(1, 15)
                    end = start + random.randint(5, 10)
                    sequence = ", ".join(str(i) for i in range(start, end + 1))
                    
                    question = question_template['question'].format(start=start, end=end)
                    correct_answer = sequence
                    
                    # Common error: skipping a number
                    if error_pattern in ['sequence_error', 'miscounting'] and random.random() < 0.7:
                        skip_pos = random.randint(0, end - start - 1) + 1
                        seq_list = list(range(start, end + 1))
                        del seq_list[skip_pos]
                        student_answer = ", ".join(str(i) for i in seq_list)
                    else:
                        student_answer = correct_answer
                else:
                    step = random.choice([2, 5, 10])
                    start = step
                    next_val = start + step
                    answer = next_val + step
                    
                    question = question_template['question'].format(
                        step=step, start=start, next=next_val
                    )
                    correct_answer = str(answer)
                    
                    # Common error: adding wrong amount
                    if error_pattern in ['calculation_error', 'sequence_error'] and random.random() < 0.7:
                        student_answer = str(next_val + step + random.choice([-1, 1, step]))
                    else:
                        student_answer = correct_answer
            
            elif q_type == 'number_sequence':
                base = random.randint(1, 10)
                step = random.randint(1, 5) * 5 // 5  # Stick to simple steps
                seq_length = random.randint(3, 4)
                sequence = ", ".join(str(base + i * step) for i in range(seq_length))
                next_val = base + seq_length * step
                
                question = question_template['question'].format(sequence=sequence)
                correct_answer = str(next_val)
                
                # Common error: adding the last difference to the first number instead of last
                if error_pattern in ['sequence_error', 'calculation_error'] and random.random() < 0.7:
                    wrong_next = base + (seq_length + 1) * step - random.choice([step, step // 2])
                    student_answer = str(wrong_next)
                else:
                    student_answer = correct_answer
            
            elif q_type == 'addition':
                if diagnosis == 'dyscalculia':
                    a = random.randint(2, 9)
                    b = random.randint(2, 9)
                else:
                    a = random.randint(4, 15)
                    b = random.randint(4, 15)
                
                sum_val = a + b
                
                question = question_template['question'].format(a=a, b=b)
                correct_answer = str(sum_val)
                
                # Common errors in addition
                if error_pattern in ['calculation_error', 'operation_confusion'] and random.random() < 0.7:
                    if random.random() < 0.5:
                        student_answer = str(sum_val + random.choice([-1, 1]))  # Off by one
                    else:
                        student_answer = str(a - b if a > b else b - a)  # Subtraction instead
                else:
                    student_answer = correct_answer
            
            elif q_type == 'subtraction':
                if diagnosis == 'dyscalculia':
                    b = random.randint(1, 5)
                    a = b + random.randint(1, 7)
                else:
                    b = random.randint(3, 9)
                    a = b + random.randint(3, 14)
                
                difference = a - b
                
                question = question_template['question'].format(a=a, b=b)
                correct_answer = str(difference)
                
                # Common errors in subtraction
                if error_pattern in ['calculation_error', 'operation_confusion', 'direction_error'] and random.random() < 0.7:
                    if random.random() < 0.3:
                        student_answer = str(difference + random.choice([-1, 1]))  # Off by one
                    elif random.random() < 0.6:
                        student_answer = str(a + b)  # Addition instead
                    else:
                        student_answer = str(b - a if b < a else a - b)  # Reversed order
                else:
                    student_answer = correct_answer
            
            elif q_type == 'multiplication':
                if diagnosis == 'dyscalculia':
                    a = random.randint(2, 5)
                    b = random.randint(2, 5)
                else:
                    a = random.randint(2, 7)
                    b = random.randint(2, 7)
                
                product = a * b
                
                question = question_template['question'].format(a=a, b=b)
                correct_answer = str(product)
                
                # Common errors in multiplication
                if error_pattern in ['calculation_error', 'operation_confusion'] and random.random() < 0.7:
                    if random.random() < 0.4:
                        student_answer = str(a + b)  # Addition instead
                    else:
                        # Common error: multiplying incorrectly
                        student_answer = str(product + random.randint(-3, 3))
                else:
                    student_answer = correct_answer
            
            elif q_type == 'place_value':
                number = random.randint(20, 99)
                ones = number % 10
                tens = number // 10
                
                if 'ones' in question_template['answer']:
                    question = question_template['question'].format(number=number)
                    correct_answer = str(ones)
                    
                    # Common error: reporting tens instead of ones
                    if error_pattern in ['place_value_confusion', 'reversal'] and random.random() < 0.7:
                        student_answer = str(tens)
                    else:
                        student_answer = correct_answer
                else:
                    question = question_template['question'].format(number=number)
                    correct_answer = str(tens)
                    
                    # Common error: reporting ones instead of tens
                    if error_pattern in ['place_value_confusion', 'reversal'] and random.random() < 0.7:
                        student_answer = str(ones)
                    else:
                        student_answer = correct_answer
            
            elif q_type == 'word_problem':
                # Simple word problem templates
                names = ['Sam', 'Maria', 'Tom', 'Lisa', 'Alex']
                objects = ['apples', 'balloons', 'stickers', 'toys', 'marbles']
                actions = ['gave away', 'lost', 'found', 'received']
                
                name = random.choice(names)
                pronoun = 'he' if name in ['Sam', 'Tom', 'Alex'] else 'she'
                object_type = random.choice(objects)
                action = random.choice(actions)
                
                if action in ['gave away', 'lost']:
                    # Subtraction problem
                    total = random.randint(5, 15)
                    change = random.randint(1, total - 1)
                    result = total - change
                else:
                    # Addition problem
                    total = random.randint(3, 10)
                    change = random.randint(1, 7)
                    result = total + change
                
                question = question_template['question'].format(
                    name=name, total=total, objects=object_type,
                    action=action, change=change, pronoun=pronoun
                )
                correct_answer = str(result)
                
                # Common errors in word problems
                if error_pattern in ['operation_confusion', 'calculation_error'] and random.random() < 0.7:
                    if action in ['gave away', 'lost'] and random.random() < 0.5:
                        student_answer = str(total + change)  # Addition instead of subtraction
                    elif action in ['found', 'received'] and random.random() < 0.5:
                        student_answer = str(max(1, total - change))  # Subtraction instead of addition
                    else:
                        student_answer = str(result + random.choice([-1, 1]))  # Off by one
                else:
                    student_answer = correct_answer
            
            else:
                # Fallback for any other question types
                question = "Generic math question"
                correct_answer = "Answer"
                student_answer = "Answer" if random.random() > 0.3 else "Wrong"
            
            # Determine if the answer is correct (for consistency)
            is_correct = 1 if student_answer == correct_answer else 0
            
            # Generate response time based on diagnosis and correctness
            if diagnosis == 'dyscalculia':
                base_time = random.uniform(8.0, 18.0)
            elif diagnosis == 'math_difficulty':
                base_time = random.uniform(6.0, 16.0)
            else:
                base_time = random.uniform(4.0, 12.0)
                
            # Incorrect answers typically take longer
            if not is_correct:
                base_time *= random.uniform(1.1, 1.3)
                
            response_time = round(base_time, 1)
            
            # Determine number of attempts (more for wrong answers)
            attempt_count = 1 if is_correct else random.randint(1, 3)
            
            # Add to the dataset
            error_data.append({
                'student_id': student_id,
                'question_id': question_id,
                'question_type': q_type,
                'question': question,
                'student_answer': student_answer,
                'correct_answer': correct_answer,
                'is_correct': is_correct,
                'response_time_seconds': response_time,
                'attempt_count': attempt_count,
                'session_date': session_date
            })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(error_data)
    filepath = save_dataset(df, output_file)
    
    print(f"Generated {len(df)} error records across {len(df['student_id'].unique())} students")
    print(f"Question type distribution:")
    print(df['question_type'].value_counts())
    
    return df

def generate_intervention_tracking_data(assessment_df, output_file='intervention_tracking_data.csv'):
    """
    Generate a simulated dataset of intervention tracking data.
    
    Parameters:
    -----------
    assessment_df : pd.DataFrame
        The assessment data to base interventions on
    output_file : str
        Name of the output CSV file
        
    Returns:
    --------
    pd.DataFrame
        The generated dataset
    """
    print("Generating intervention tracking data...")
    
    # Intervention types
    intervention_types = [
        'multisensory_approach', 'visual_aids', 'game_based_learning', 
        'structured_sequence', 'technology_assisted'
    ]
    
    # Get students with dyscalculia or math difficulties
    students_with_difficulties = assessment_df[
        assessment_df['diagnosis'].isin(['dyscalculia', 'math_difficulty'])
    ]['student_id'].tolist()
    
    # Create a list to hold the data
    intervention_data = []
    intervention_id_counter = 1
    
    for student_id in students_with_difficulties:
        # Get student details
        student = assessment_df[assessment_df['student_id'] == student_id].iloc[0]
        diagnosis = student['diagnosis']
        
        # Determine the number of interventions (1-2)
        num_interventions = random.choices([1, 2], weights=[0.4, 0.6])[0]
        
        # Select intervention types based on student's profile
        # For example, focus on specific interventions based on error patterns
        error_pattern = student['error_patterns']
        
        # Determine good intervention matches based on error patterns
        if error_pattern in ['transposition', 'reversal']:
            primary_interventions = ['multisensory_approach', 'visual_aids']
        elif error_pattern in ['miscounting', 'sequence_error']:
            primary_interventions = ['structured_sequence', 'game_based_learning']
        elif error_pattern in ['operation_confusion']:
            primary_interventions = ['visual_aids', 'technology_assisted']
        elif error_pattern in ['multiple_errors']:
            primary_interventions = ['multisensory_approach', 'structured_sequence']
        else:
            primary_interventions = random.sample(intervention_types, 2)
        
        # Ensure we have enough interventions to choose from
        available_interventions = primary_interventions + [
            i for i in intervention_types if i not in primary_interventions
        ]
        
        selected_interventions = []
        for _ in range(num_interventions):
            if available_interventions:
                intervention = available_interventions.pop(0)
                selected_interventions.append(intervention)
        
        # For each intervention, create an entry
        start_date = datetime.datetime(2024, 9, 1)
        for intervention_idx, intervention_type in enumerate(selected_interventions):
            intervention_id = f"INT{intervention_id_counter:03d}"
            intervention_id_counter += 1
            
            # Each intervention lasts 6 weeks
            duration_weeks = 6
            if intervention_idx == 0:
                # First intervention starts at a random date in September
                start = start_date + datetime.timedelta(days=random.randint(0, 30))
            else:
                # Subsequent interventions start after the previous one
                previous_end_date = datetime.datetime.strptime(
                    intervention_data[-1]['end_date'], '%Y-%m-%d'
                )
                start = previous_end_date + datetime.timedelta(days=random.randint(1, 7))
            
            end = start + datetime.timedelta(weeks=duration_weeks)
            
            # Format dates as strings
            start_date_str = start.strftime('%Y-%m-%d')
            end_date_str = end.strftime('%Y-%m-%d')
            
            # Number of sessions completed (out of 12 possible)
            sessions_completed = random.randint(10, 12)
            
            # Pre and post assessment scores
            # Base the pre score on the student's profile
            difficulty_score = sum([
                student['number_recognition'], student['number_comparison'],
                student['counting_skills'], student['place_value'],
                student['calculation_accuracy'], student['word_problem_solving']
            ])
            
            # Normalize to a 0-100 scale (6 metrics, each 1-5)
            pre_score = int((difficulty_score / 30) * 100)
            
            # Improvement depends on diagnosis and intervention match
            if diagnosis == 'dyscalculia':
                if intervention_type in primary_interventions:
                    improvement_base = random.randint(4, 7)
                else:
                    improvement_base = random.randint(2, 5)
            else:  # math_difficulty
                if intervention_type in primary_interventions:
                    improvement_base = random.randint(5, 8)
                else:
                    improvement_base = random.randint(3, 6)
            
            # Add some randomness to the improvement
            improvement = improvement_base + random.randint(-1, 1)
            
            # Ensure the post score doesn't exceed 100
            post_score = min(100, pre_score + improvement)
            
            # Individual skill improvements
            # More improvement in areas targeted by the intervention
            if intervention_type == 'multisensory_approach':
                number_recog_imp = random.randint(1, 2)
                number_comp_imp = random.randint(1, 2)
                counting_imp = random.randint(0, 1)
                calculation_imp = random.randint(1, 2)
                problem_solving_imp = random.randint(0, 1)
            elif intervention_type == 'visual_aids':
                number_recog_imp = random.randint(1, 2)
                number_comp_imp = random.randint(1, 2)
                counting_imp = random.randint(0, 1)
                calculation_imp = random.randint(0, 1)
                problem_solving_imp = random.randint(1, 2)
            elif intervention_type == 'game_based_learning':
                number_recog_imp = random.randint(0, 1)
                number_comp_imp = random.randint(1, 2)
                counting_imp = random.randint(1, 2)
                calculation_imp = random.randint(1, 2)
                problem_solving_imp = random.randint(0, 1)
            elif intervention_type == 'structured_sequence':
                number_recog_imp = random.randint(1, 2)
                number_comp_imp = random.randint(0, 1)
                counting_imp = random.randint(1, 2)
                calculation_imp = random.randint(0, 1)
                problem_solving_imp = random.randint(0, 1)
            else:  # technology_assisted
                number_recog_imp = random.randint(0, 1)
                number_comp_imp = random.randint(1, 2)
                counting_imp = random.randint(0, 1)
                calculation_imp = random.randint(1, 2)
                problem_solving_imp = random.randint(1, 2)
            
            # Math anxiety usually decreases with intervention
            math_anxiety_change = 'decreased'
            
            # Generate feedback
            teacher_feedback_options = [
                "Student shows improved number recognition but still struggles with calculations",
                "Visual representations helped with conceptual understanding",
                "More engaged in sessions, shows improved confidence",
                "Needs consistent routine, making slow progress",
                "Responds well to interactive digital tools",
                "Improved manipulation of concrete materials",
                "Needs very small steps and lots of repetition",
                "Visual number lines have been particularly helpful",
                "High engagement with game formats",
                "Digital tools help maintain focus",
                "Touch and movement help reinforce concepts",
                "Very small sequential steps showing progress",
                "Color-coding place values has been effective"
            ]
            
            parent_feedback_options = [
                "Child is less anxious about math homework",
                "Homework time is less stressful",
                "Practices math with siblings now",
                "Still reluctant but less tearful",
                "Uses math apps at home voluntarily",
                "More willing to try new problems",
                "Starting to count objects at home",
                "Less frustration when practicing",
                "Talks about math concepts at home now",
                "Asks to use math apps at home",
                "Reports math is 'not as boring'",
                "Still anxious but willing to try",
                "Uses drawing to solve problems now"
            ]
            
            teacher_feedback = random.choice(teacher_feedback_options)
            parent_feedback = random.choice(parent_feedback_options)
            
            intervention_data.append({
                'student_id': student_id,
                'intervention_id': intervention_id,
                'intervention_type': intervention_type,
                'start_date': start_date_str,
                'end_date': end_date_str,
                'duration_weeks': duration_weeks,
                'sessions_completed': sessions_completed,
                'pre_assessment_score': pre_score,
                'post_assessment_score': post_score,
                'number_recognition_improvement': number_recog_imp,
                'number_comparison_improvement': number_comp_imp,
                'counting_improvement': counting_imp,
                'calculation_improvement': calculation_imp,
                'problem_solving_improvement': problem_solving_imp,
                'math_anxiety_change': math_anxiety_change,
                'teacher_feedback': teacher_feedback,
                'parent_feedback': parent_feedback
            })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(intervention_data)
    filepath = save_dataset(df, output_file)
    
    print(f"Generated {len(df)} intervention records for {len(df['student_id'].unique())} students")
    print(f"Intervention type distribution:")
    print(df['intervention_type'].value_counts())
    
    return df

def main():
    """Main function to generate all datasets."""
    parser = argparse.ArgumentParser(description='Generate datasets for the MathemAI project')
    parser.add_argument('--n', type=int, default=100, help='Number of assessment records to generate')
    parser.add_argument('--output-dir', type=str, default='../datasets', help='Output directory for datasets')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    create_directory_if_not_exists(args.output_dir)
    
    # Generate assessment data
    assessment_df = generate_dyscalculia_assessment_data(
        n=args.n, 
        output_file=os.path.join(args.output_dir, 'dyscalculia_assessment_data.csv')
    )
    
    # Generate error analysis data based on assessment data
    error_df = generate_error_analysis_data(
        assessment_df, 
        output_file=os.path.join(args.output_dir, 'error_analysis_data.csv')
    )
    
    # Generate intervention tracking data based on assessment data
    intervention_df = generate_intervention_tracking_data(
        assessment_df, 
        output_file=os.path.join(args.output_dir, 'intervention_tracking_data.csv')
    )
    
    print("\nDataset generation complete!")
    print(f"All datasets saved to directory: {args.output_dir}")

if __name__ == "__main__":
    main()