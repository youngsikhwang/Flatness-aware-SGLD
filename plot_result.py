import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(file_path):
    """Load training results from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def plot_individual_results(data, optimizer_name, save_dir='plots'):
    """Plot train_acc and test_acc for individual optimizer."""
    results = data['results']
    epochs = [r['epoch'] for r in results]
    train_accs = [r['train_acc'] for r in results]
    test_accs = [r['test_acc'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, 'b-', label='Train Accuracy', marker='o', markersize=4)
    plt.plot(epochs, test_accs, 'r-', label='Test Accuracy', marker='s', markersize=4)
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{optimizer_name.upper()} - Training vs Test Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Display final performance info
    final_train = train_accs[-1] if train_accs else 0
    final_test = test_accs[-1] if test_accs else 0
    best_test = data.get('best_test_acc', 0)
    
    plt.text(0.02, 0.98, f'Final Train: {final_train:.2f}%\nFinal Test: {final_test:.2f}%\nBest Test: {best_test:.2f}%', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save the plot
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / f'{optimizer_name}_individual_results.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"개별 결과 그래프 저장: {save_path}")
    
    plt.show()

def plot_comparison(data_dict, save_dir='plots'):
    """Plot comparison of test_acc across multiple optimizers."""
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, (optimizer_name, data) in enumerate(data_dict.items()):
        results = data['results']
        epochs = [r['epoch'] for r in results]
        test_accs = [r['test_acc'] for r in results]
        
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        plt.plot(epochs, test_accs, color=color, marker=marker, 
                label=f'{optimizer_name.upper()}', linewidth=2, markersize=6)
    
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy Comparison Across Optimizers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Display final performance summary
    summary_text = "Final Performance:\n"
    for optimizer_name, data in data_dict.items():
        final_test = data['results'][-1]['test_acc'] if data['results'] else 0
        best_test = data.get('best_test_acc', 0)
        summary_text += f"{optimizer_name.upper()}: {final_test:.2f}% (best: {best_test:.2f}%)\n"
    
    plt.text(0.02, 0.98, summary_text.strip(), 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Save comparison plot
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / 'optimizer_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"비교 그래프 저장: {save_path}")
    
    plt.show()

def main():
    """Main function"""
    # Set JSON file paths (modify as needed)
    file_paths = {
        'sgd': '/mountpoint/ajh/fgld/code/optuna_results/cifar10N_resnet34_sgd/final_training_results_sgd.json',      # SGD results file path
        'sam': '/mountpoint/ajh/fgld/code/optuna_results/cifar10N_resnet34_sam/final_training_results_sam.json',      # SAM results file path  
        'fgld': '/mountpoint/ajh/fgld/code/optuna_results/cifar10N_resnet34_fgld/final_training_results_fgld.json'     # FGLD results file path
    }
    
    # Set plot save directory
    save_dir = 'plots'
    
    # Load data
    data_dict = {}
    
    for optimizer_name, file_path in file_paths.items():
        try:
            data = load_results(file_path)
            data_dict[optimizer_name] = data
            print(f"{optimizer_name.upper()} 데이터 로드 완료: {len(data['results'])} epochs")
        except FileNotFoundError:
            print(f"Warning: {file_path} 파일을 찾을 수 없습니다. {optimizer_name.upper()} 데이터를 건너뜁니다.")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not data_dict:
        print("로드된 데이터가 없습니다. 파일 경로를 확인해주세요.")
        return
    
    # Create plots directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    print(f"그래프 저장 디렉토리: {Path(save_dir).absolute()}")
    
    # 1. Plot individual optimizer results
    print("\n개별 옵티마이저 결과 그래프를 생성합니다...")
    for optimizer_name, data in data_dict.items():
        plot_individual_results(data, optimizer_name, save_dir)
    
    # 2. Plot optimizer comparison
    if len(data_dict) > 1:
        print("\n옵티마이저 비교 그래프를 생성합니다...")
        plot_comparison(data_dict, save_dir)
    
    # 3. Print performance summary
    print("\n=== 성능 요약 ===")
    for optimizer_name, data in data_dict.items():
        results = data['results']
        if results:
            final_train = results[-1]['train_acc']
            final_test = results[-1]['test_acc']
            best_test = data.get('best_test_acc', final_test)
            avg_last_test = data.get('avg_last_test_acc', final_test)
            
            print(f"{optimizer_name.upper()}:")
            print(f"  Final Train Accuracy: {final_train:.2f}%")
            print(f"  Final Test Accuracy: {final_test:.2f}%")
            print(f"  Best Test Accuracy: {best_test:.2f}%")
            print(f"  Avg Last Test Accuracy: {avg_last_test:.3f}%")
            print()
    
    print(f"\n모든 그래프가 '{save_dir}' 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main()