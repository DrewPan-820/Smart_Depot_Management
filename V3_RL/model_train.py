from V3_RL.train.train_dqn import train_dqn
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    NUM_STACKS = 5
    STACK_HEIGHT = 4
    NUM_ORDERS = 30

    print(f"\n🚀 Starting Attention DQN training with wait action and num_stacks={NUM_STACKS}...")

    # 返回最优保存信息
    rewards, agent, best_episode, best_reward = train_dqn(
        episodes=500,
        stack_height=STACK_HEIGHT,
        num_orders=NUM_ORDERS
    )

    print("\n✅ Training completed.")
    print(f"📈 Final episode reward: {rewards[-1]:.2f}")
    print(f"🏅 Best model was saved at episode {best_episode}, reward={best_reward:.2f}")
    print("💾 Best model: trained_dqn_best.pth")
    print("💾 Latest model: trained_dqn.pth")

    # 绘制训练奖励曲线
    plt.plot(rewards, label="Reward per Episode")
    plt.title(f"Reward Trend of Attention DQN with Wait (num_stacks={NUM_STACKS})")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.show()
