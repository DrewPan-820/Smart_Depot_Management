from V3_RL.train.train_dqn import train_dqn_with_wait
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    NUM_STACKS = 5  # 明确指定仓库堆栈数量，确保模型适配
    STACK_HEIGHT = 4
    NUM_ORDERS = 30

    print(f"\n🚀 Starting Attention DQN training with wait action and num_stacks={NUM_STACKS}...")

    # 调用更新后的训练函数
    rewards, agent = train_dqn_with_wait(
        episodes=1000,
        stack_height=4
    )

    print("\n✅ Training completed.")
    print(f"📈 Final episode reward: {rewards[-1]:.2f}")

    # 保存模型
    torch.save(agent.policy_net.state_dict(), "trained_dqn.pth")
    print("💾 Model saved as trained_dqn.pth")

    # 绘制训练奖励曲线
    plt.plot(rewards, label="Reward per Episode")
    plt.title(f"Reward Trend of Attention DQN with Wait (num_stacks={NUM_STACKS})")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.show()
