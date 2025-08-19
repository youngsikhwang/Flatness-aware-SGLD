export CUDA_VISIBLE_DEVICES=3

echo "Run Start"

# ========== 실험 파라미터 설정 ==========
N_TRIALS=50          # Optuna trial 수
EPOCHS=200       # 각 trial의 epochs
FINAL_EPOCHS=200     # 최종 학습 epochs
# =====================================================

# 데이터셋 및 백본 설정
DATASETS=("cifar10N" "cifar100N")
BACKBONES=("resnet50")
OPTIMIZERS=("sgd" "sam" "sgld" "fgld") 

START_TIME=$(date +%s)
echo "=========================================="
echo "Starting All Optimizer Experiments"
echo "Time: $(date)"
echo "Settings: N_TRIALS=$N_TRIALS, EPOCHS=$EPOCHS, FINAL_EPOCHS=$FINAL_EPOCHS"
echo "=========================================="

# 각 조합에 대해 실행
for DATASET in "${DATASETS[@]}"; do
    for BACKBONE in "${BACKBONES[@]}"; do
        for OPTIMIZER in "${OPTIMIZERS[@]}"; do
            
            echo ""
            echo "=========================================="
            echo "Dataset: $DATASET | Backbone: $BACKBONE | Optimizer: $OPTIMIZER"
            echo "=========================================="
            
            # 실험별 저장 디렉토리
            SAVE_DIR="./optuna_results/${DATASET}_${BACKBONE}_${OPTIMIZER}"
            
            echo "Starting optimization for $OPTIMIZER..."
            echo "Results will be saved to: $SAVE_DIR"
            
            # Python 명령 실행
            python main_auto.py \
                --dataset $DATASET \
                --backbone $BACKBONE \
                --optimizer $OPTIMIZER \
                --n_trials $N_TRIALS \
                --epochs $EPOCHS \
                --final_epochs $FINAL_EPOCHS \
                --train_final \
                --visualize \
                --save_dir $SAVE_DIR
            
            if [ $? -eq 0 ]; then
                echo "✓ Successfully completed: $DATASET + $BACKBONE + $OPTIMIZER"
            else
                echo "✗ Failed: $DATASET + $BACKBONE + $OPTIMIZER"
            fi
            
            # GPU 메모리 정리
            sleep 5
            
        done
    done
done

# 종료 시간 계산
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Total time: ${HOURS}h ${MINUTES}m"
echo "=========================================="

