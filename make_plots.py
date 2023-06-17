
import matplotlib.pyplot as plt

"""
 compare on:
        pop size - V
        elites?
        network sizes - V
        threshold in predict
        diff activation functions
"""


def plot_fitness_vs_population(pop_300, pop_200, pop_150, pop_100,pop_50):
    plt.figure(figsize=(12, 8))
    # Ensure that x-axis values (generations) align with the length of each score list
    generations_run1 = range(len(pop_300))
    generations_run2 = range(len(pop_200))
    generations_run3 = range(len(pop_150))
    generations_run4 = range(len(pop_100))
    generations_run5 = range(len(pop_50))

    # Plotting each run's data
    plt.plot(generations_run1, pop_300, label='Population 300')
    plt.plot(generations_run2, pop_200, label='Population 200')
    plt.plot(generations_run3, pop_150, label='Population 150')
    plt.plot(generations_run4, pop_100, label='Population 100')
    plt.plot(generations_run5, pop_50, label='Population 50')

    # Annotate the last point in each line
    plt.annotate(f'{pop_300[-1]:.2f}', (generations_run1[-1], pop_300[-1]))
    plt.annotate(f'{pop_200[-1]:.2f}', (generations_run2[-1], pop_200[-1]))
    plt.annotate(f'{pop_150[-1]:.2f}', (generations_run3[-1], pop_150[-1]))
    plt.annotate(f'{pop_100[-1]:.2f}', (generations_run4[-1], pop_100[-1]))
    plt.annotate(f'{pop_50[-1]:.2f}', (generations_run5[-1], pop_50[-1]))

    # Providing labels for better readability
    plt.xlabel('Generation', fontsize=16)
    plt.ylabel('Best Fitness', fontsize=16)
    plt.title('Best Fitness VS. Different Population Size', fontsize=16)
    plt.legend()
    plt.show()


def plot_fitness_vs_NN0_params(hiddens_16, hiddens_32, hiddens_64, hiddens_128):
    plt.figure(figsize=(12, 8))

    # Ensure that x-axis values (generations) align with the length of each score list
    generations_run1 = range(len(hiddens_16))
    generations_run2 = range(len(hiddens_32))
    generations_run3 = range(len(hiddens_64))
    generations_run4 = range(len(hiddens_128))

    # Plotting each run's data
    plt.plot(generations_run1, hiddens_16, label='hiddens_16')
    plt.plot(generations_run2, hiddens_32, label='hiddens_32')
    plt.plot(generations_run3, hiddens_64, label='hiddens_64')
    plt.plot(generations_run4, hiddens_128, label='hiddens_128')

    # Annotate the last point in each line
    plt.annotate(f'{hiddens_16[-1]:.2f}', (generations_run1[-1], hiddens_16[-1]))
    plt.annotate(f'{hiddens_32[-1]:.2f}', (generations_run2[-1], hiddens_32[-1]))
    plt.annotate(f'{hiddens_64[-1]:.2f}', (generations_run3[-1], hiddens_64[-1]))
    plt.annotate(f'{hiddens_128[-1]:.2f}', (generations_run4[-1], hiddens_128[-1]))

    # Providing labels for better readability
    plt.xlabel('Generation', fontsize=16)
    plt.ylabel('Best Fitness', fontsize=16)
    plt.title('Best Fitness VS. Different Neural Network Parameters', fontsize=16)
    plt.legend()
    plt.show()


def plot_fitness_vs_NN1_params(h_16_64_32_32, h_16_16_16_16, h_16_32_32_32, h_16_64_64_64, h_16_128_128_128):
    plt.figure(figsize=(12, 8))

    # Ensure that x-axis values (generations) align with the length of each score list
    generations_run1 = range(len(h_16_64_32_32))
    generations_run2 = range(len(h_16_16_16_16))
    generations_run3 = range(len(h_16_32_32_32))
    generations_run4 = range(len(h_16_64_64_64))
    generations_run5 = range(len(h_16_128_128_128))

    # Plotting each run's data
    plt.plot(generations_run1, h_16_64_32_32, label='h_16_64_32_32')
    plt.plot(generations_run2, h_16_16_16_16, label='h_16_16_16_16')
    plt.plot(generations_run3, h_16_32_32_32, label='h_16_32_32_32')
    plt.plot(generations_run4, h_16_64_64_64, label='h_16_64_64_64')
    plt.plot(generations_run5, h_16_128_128_128, label='h_16_128_128_128')

    # Annotate the last point in each line
    plt.annotate(f'{h_16_64_32_32[-1]:.2f}', (generations_run1[-1], h_16_64_32_32[-1]))
    plt.annotate(f'{h_16_16_16_16[-1]:.2f}', (generations_run2[-1], h_16_16_16_16[-1]))
    plt.annotate(f'{h_16_32_32_32[-1]:.2f}', (generations_run3[-1], h_16_32_32_32[-1]))
    plt.annotate(f'{h_16_64_64_64[-1]:.2f}', (generations_run4[-1], h_16_64_64_64[-1]))
    plt.annotate(f'{h_16_128_128_128[-1]:.2f}', (generations_run5[-1], h_16_128_128_128[-1]))

    # Providing labels for better readability
    plt.xlabel('Generation', fontsize=16)
    plt.ylabel('Best Fitness', fontsize=16)
    plt.title('Best Fitness VS. Different Neural Network Parameters', fontsize=16)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    pop_300 = [0.69556, 0.69556, 0.69556, 0.69556, 0.70794, 0.81737, 0.81881, 0.81881, 0.81881, 0.83106, 0.91275, 0.91275,
     0.91275, 0.91275, 0.92775, 0.95556, 0.97294, 0.97575, 0.98831, 0.98938, 0.98938, 0.98938, 0.98938, 0.98938,
     0.98938, 0.98938, 0.98938, 0.98938, 0.98938, 0.98938, 0.98938, 0.98938, 0.98938, 0.98938, 0.98938, 0.98938,
     0.98938, 0.98938, 0.98938, 0.98938, 0.989375]
    pop_200 = [0.58725, 0.58725, 0.61013, 0.62362, 0.67031, 0.69269, 0.80375, 0.80375, 0.83969, 0.83969, 0.88225, 0.91863, 0.93344, 0.94481, 0.95331, 0.97925, 0.98562, 0.98631, 0.98919, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895]
    pop_150 = [0.72419, 0.72419, 0.72419, 0.72419, 0.73638, 0.82488, 0.82488, 0.82488, 0.88806, 0.90631, 0.907, 0.94581, 0.95125, 0.95431, 0.98388, 0.98388, 0.98388, 0.98975, 0.98975, 0.98975, 0.98975, 0.98975, 0.98975, 0.98975, 0.98975, 0.98975, 0.98975, 0.98975, 0.98975, 0.98975, 0.98975, 0.98975, 0.98975, 0.98975, 0.98975, 0.98975, 0.98975, 0.98975, 0.98975]
    pop_100 = [0.51562, 0.51562, 0.68544, 0.68544, 0.68544, 0.74744, 0.77463, 0.82369, 0.82369, 0.84962, 0.88931, 0.89819, 0.94388, 0.96206, 0.96506, 0.97188, 0.97319, 0.98487, 0.98794, 0.98894, 0.98894, 0.98894, 0.98894, 0.98894, 0.98894, 0.98894, 0.98894, 0.98894, 0.98894, 0.98894, 0.98894, 0.98894, 0.98894, 0.98894, 0.98894, 0.98894, 0.98894, 0.98894, 0.98894, 0.98894, 0.9889375]
    pop_50 = [0.48519, 0.521, 0.521, 0.57775, 0.69369, 0.82112, 0.82225, 0.89044, 0.89044, 0.89781, 0.91031, 0.92556, 0.93288, 0.95625, 0.96881, 0.97781, 0.97781, 0.97781, 0.98681, 0.98681, 0.98681, 0.98719, 0.98787, 0.989, 0.989, 0.989, 0.989, 0.989, 0.989, 0.989, 0.989, 0.989, 0.989, 0.989, 0.989, 0.989, 0.989, 0.989, 0.989, 0.989, 0.989, 0.989, 0.989, 0.989, 0.989]
    # plot_fitness_vs_population(pop_300, pop_200, pop_150, pop_100, pop_50)

    # all tested on pop 200:
    # NN_0:
    hiddens_16 = [0.58725, 0.58725, 0.61013, 0.62362, 0.67031, 0.69269, 0.80375, 0.80375, 0.83969, 0.83969, 0.88225, 0.91863, 0.93344, 0.94481, 0.95331, 0.97925, 0.98562, 0.98631, 0.98919, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895, 0.9895]
    hiddens_32 = [0.67706, 0.67706, 0.67706, 0.71344, 0.71344, 0.72213, 0.7395, 0.79019, 0.79019, 0.79056, 0.806, 0.859, 0.88738, 0.90338, 0.92256, 0.94481, 0.95669, 0.9715, 0.97819, 0.98581, 0.98881, 0.98881, 0.98881, 0.98881, 0.98881, 0.98881, 0.98881, 0.98881, 0.98881, 0.98881, 0.98881, 0.98881, 0.98881, 0.98881, 0.98881, 0.98881, 0.98881, 0.98881, 0.98881, 0.98881, 0.98881, 0.9888125]
    hiddens_64 = [0.48594, 0.48594, 0.48594, 0.48594, 0.55131, 0.55131, 0.55131, 0.60688, 0.6145, 0.75844, 0.75844, 0.75844, 0.75844, 0.75844, 0.88025, 0.88025, 0.9005, 0.93588, 0.95462, 0.97569, 0.98556, 0.988, 0.98862, 0.98862, 0.98862, 0.98862, 0.98862, 0.98862, 0.98862, 0.98862, 0.98862, 0.98862, 0.98862, 0.98862, 0.98862, 0.98862, 0.98862, 0.98862, 0.98862, 0.98862, 0.98862, 0.98862, 0.98862, 0.988625]
    hiddens_128 = [0.49506, 0.49506, 0.49506, 0.49506, 0.49506, 0.76138, 0.76138, 0.76138, 0.76138, 0.76138, 0.87938, 0.87938, 0.91806, 0.91806, 0.91806, 0.96137, 0.96137, 0.97288, 0.97437, 0.98306, 0.988, 0.98875, 0.98875, 0.98875, 0.98875, 0.98875, 0.98875, 0.98875, 0.98875, 0.98875, 0.98875, 0.98875, 0.98875, 0.98875, 0.98875, 0.98875, 0.98875, 0.98875, 0.98875, 0.98875, 0.98875, 0.98875, 0.98875]
    # plot_fitness_vs_NN0_params(hiddens_16, hiddens_32, hiddens_64, hiddens_128)

    # NN_1:
    h_16_64_32_32 = [0.70275, 0.70275, 0.70275, 0.72013, 0.72013, 0.72013, 0.72013, 0.72013, 0.84106, 0.84106, 0.84744, 0.84744, 0.84744, 0.84744, 0.85987, 0.91463, 0.91463, 0.92106, 0.93363, 0.94425, 0.95806, 0.96, 0.97806, 0.989, 0.99262, 0.99744, 0.99919, 0.99962, 1.0, 1.0]
    h_16_16_16_16 = [0.72169, 0.774, 0.774, 0.774, 0.78212, 0.78212, 0.80994, 0.86656, 0.86656, 0.88681, 0.89231, 0.92763, 0.92763, 0.95581, 0.96681, 0.96681, 0.96681, 0.97194, 0.9845, 0.99269, 0.99269, 0.99675, 0.99975, 0.99975, 0.99987, 0.99987, 1.0, 1.0]
    h_16_32_32_32 = [0.56975, 0.56975, 0.56975, 0.56975, 0.56975, 0.74619, 0.74619, 0.74619, 0.74619, 0.74619, 0.83406, 0.85525, 0.85525, 0.85525, 0.85525, 0.88356, 0.88356, 0.91363, 0.91363, 0.91363, 0.94544, 0.94544, 0.95581, 0.97537, 0.97537, 0.97769, 0.98538, 0.99219, 0.99869, 0.9995, 0.99969, 1.0, 1.0]
    h_16_64_64_64 = [0.71275, 0.71275, 0.71275, 0.71275, 0.71275, 0.74069, 0.74069, 0.74069, 0.74069, 0.74069, 0.79625, 0.84962, 0.85194, 0.85719, 0.90719, 0.91638, 0.91638, 0.93469, 0.94563, 0.955, 0.96344, 0.9675, 0.98319, 0.98388, 0.99331, 0.99581, 0.99919, 0.99962, 1.0, 1.0]
    h_16_128_128_128 = [0.63013, 0.63013, 0.63013, 0.63013, 0.63013, 0.80369, 0.80369, 0.80369, 0.80369, 0.80369, 0.82131, 0.82131, 0.82131, 0.82131, 0.82131, 0.87994, 0.88575, 0.88575, 0.91475, 0.92856, 0.94169, 0.95625, 0.95969, 0.98175, 0.98894, 0.98894, 0.99762, 0.99831, 0.99969, 0.99969, 0.99987, 1.0, 1.0]
    plot_fitness_vs_NN1_params(h_16_64_32_32, h_16_16_16_16, h_16_32_32_32, h_16_64_64_64, h_16_128_128_128)

