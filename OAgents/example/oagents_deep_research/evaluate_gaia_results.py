import argparse
import pandas as pd
from scripts.gaia_scorer import question_scorer, check_close_call
from pathlib import Path

def evaluate_results(jsonl_file):    
    if not Path(jsonl_file).exists():
        print(f"Error: File {jsonl_file} not found!")
        return
    
    # JSONL 파일 로드
    print(f"Loading results from: {jsonl_file}")
    try:
        df = pd.read_json(jsonl_file, lines=True)
        print(f"Loaded {len(df)} results")
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # 각 답변 스코어링
    results = []
    close_calls = []
    
    for idx, row in df.iterrows():
        model_answer = row.get('prediction', '')
        ground_truth = row.get('true_answer', '')
        task_id = row.get('task_id', f'task_{idx}')
        question = row.get('question', '')
        
        if model_answer is None:
            # 에러가 발생한 경우
            is_correct = False
            is_close = False
        else:
            is_correct = question_scorer(str(model_answer), str(ground_truth))
            is_close = check_close_call(str(model_answer), str(ground_truth), is_correct)
        
        results.append({
            'task_id': task_id,
            'correct': is_correct,
            'close_call': is_close and not is_correct,
            'prediction': model_answer,
            'true_answer': ground_truth,
            'question': question[:100] + '...' if len(str(question)) > 100 else question
        })
        
        if is_close and not is_correct:
            close_calls.append({
                'task_id': task_id,
                'prediction': model_answer,
                'true_answer': ground_truth
            })
    
    # 결과 분석
    results_df = pd.DataFrame(results)
    total_tasks = len(results_df)
    correct_count = results_df['correct'].sum()
    close_call_count = results_df['close_call'].sum()
    error_count = sum(1 for r in results if r['prediction'] is None)
    
    accuracy = correct_count / total_tasks if total_tasks > 0 else 0
    close_call_rate = close_call_count / total_tasks if total_tasks > 0 else 0
    
    # 결과 출력
    print("\n" + "="*60)
    print("GAIA EVALUATION RESULTS")
    print("="*60)
    print(f"Total tasks: {total_tasks}")
    print(f"Correct answers: {correct_count}")
    print(f"Close calls: {close_call_count}")
    print(f"Errors/No answer: {error_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Close call rate: {close_call_rate:.2%}")
    print("="*60)
    
    # 틀린 답변들 표시
    incorrect_results = results_df[~results_df['correct']]
    if len(incorrect_results) > 0:
        print(f"\nINCORRECT ANSWERS ({len(incorrect_results)} tasks):")
        print("-" * 60)
        for idx, row in incorrect_results.iterrows():
            print(f"Task ID: {row['task_id']}")
            print(f"Question: {row['question']}")
            print(f"Prediction: {row['prediction']}")
            print(f"True Answer: {row['true_answer']}")
            if row['close_call']:
                print("⚠️  Close call - almost correct!")
            print("-" * 60)
    
    # Close calls 따로 표시
    if close_calls:
        print(f"\nCLOSE CALLS ({len(close_calls)} tasks):")
        print("-" * 40)
        for cc in close_calls:
            print(f"Task: {cc['task_id']}")
            print(f"Predicted: {cc['prediction']}")
            print(f"Expected: {cc['true_answer']}")
            print("-" * 40)
    
    return accuracy, results_df

def main():
    parser = argparse.ArgumentParser(description='Evaluate GAIA results')
    parser.add_argument('jsonl_file', help='Path to the JSONL results file')
    args = parser.parse_args()
    
    evaluate_results(args.jsonl_file)

if __name__ == "__main__":
    main()