using System.Collections.Generic;

//Input "1" matrix for training.
int[][] inputTrainingForOne = new int[][]
{
    new int[] { 0, 0, 1, 0, 0 },
    new int[] { 0, 1, 1, 0, 0 },
    new int[] { 0, 0, 1, 0, 0 },
    new int[] { 0, 0, 1, 0, 0 },
    new int[] { 0, 0, 1, 0, 0 }
};

//The list to keep all training matrices.
List<int[][]> trainingOne = new List<int[][]>();

//First training matrix is a "1" with no base. We create it out of the for loop.
int[][] noBase = CloneArray(inputTrainingForOne);
for (int col = 0; col < 5; col++)
{
    noBase[4][col] = 0;
}
trainingOne.Add(noBase);

int totalCombinations = 16; //Total combinations for 4 pixels. (Either 1 or 0).

for (int i = 0; i < totalCombinations; i++)
{
    int[][] newOne = CloneArray(inputTrainingForOne);

    int[] combination = GetCombination(i, 4); //Get binary number of i for 4 bits. So we can create the combinations of "1"s base.

    int colIndex = 0;
    for (int col = 0; col < 5; col++)
    {
        if (col == 2) continue; //Middle 1 should be always 1 to connect "1"'s base to its middle part.

        // Set the value based on the combination.
        newOne[4][col] = combination[colIndex];
        colIndex++;
    }

    //Check to see if there is a disconnection in base.
    if ((newOne[4][0] == 0 || newOne[4][1] == 1) &&
        (newOne[4][4] == 0 || newOne[4][3] == 1))
    {
        //Check if there is another same matrix in list.
        bool isUnique = true;
        foreach (var existingOne in trainingOne)
        {
            if (AreArraysEqual(existingOne, newOne))
            {
                isUnique = false;
                break;
            }
        }

        if (isUnique)
        {
            trainingOne.Add(newOne);
        }
    }
}

//Input "2" Training Matrix.
int[][] inputTrainingForTwo = new int[][]
{
    new int[] {0,1,1,1,0},
    new int[] {1,0,0,0,1},
    new int[] {0,0,0,1,0},
    new int[] {0,0,1,0,0},
    new int[] {1,1,1,1,1}
};

//The list to keep all training matrices.
List<int[][]> trainingTwo = new List<int[][]>();

List<(int, int)> changeablePixels = FindChangeablePixels(inputTrainingForTwo);

for (int i = 0; i < 10; i++) //Create 10 variations.
{
    int[][] newTwo = CloneArray(inputTrainingForTwo);

    //Select the changeable pixel and change it.
    var pixel = changeablePixels[i % changeablePixels.Count];
    newTwo[pixel.Item1][pixel.Item2] = 1;

    trainingTwo.Add(newTwo);
}

int[][] trainingDataForOne = ChangeDataTypeOfTrainingData(trainingOne);
int[][] trainingDataForTwo = ChangeDataTypeOfTrainingData(trainingTwo);

// Keeping our training sets in a list.
List<int[][]> trainingSets = new List<int[][]>
{
    trainingDataForOne, trainingDataForTwo
};

// Creating our test sets.
int[][] testDataOneMatrix = new int[][]
{
    new int[] {0,1,0,0,0,
               1,1,0,0,0,
               0,1,0,0,0,
               0,1,0,0,0,
               1,1,1,1,0 },

    new int[] {0,0,0,0,1,
               0,0,0,1,1,
               0,0,0,0,1,
               0,0,0,0,1,
               0,0,0,0,0 }
};

int[][] testDataTwoMatrix = new int[][]
{
    new int[] {0,0,0,0,0,
               0,1,1,0,0,
               1,0,0,1,0,
               0,0,1,0,0,
               1,1,1,1,1 }
};

// Keeping our test sets in a list.
List<int[][]> testSets = new List<int[][]>()
{
    testDataOneMatrix, testDataTwoMatrix
};

// Creating our neural network by giving output counts and input counts.
NeuralNetwork neuralNetwork = new NeuralNetwork(2, 25);

// Training neural network.
neuralNetwork.Train(0.03, 40, trainingSets);

// Showing final results after training.
neuralNetwork.ShowFinalResults(trainingSets);

// Showing results for a different data set.
Console.WriteLine("\nEğitim setlerinden farklı olarak verdiğimiz set için tahmin değerleri: \n");
neuralNetwork.ShowFinalResults(testSets);


Console.ReadKey();


// Get 4 bits binary numbers to create 4 pixel combinations.
static int[] GetCombination(int index, int length)
{
    int[] combination = new int[length];
    for (int i = length - 1; i >= 0; i--)
    {
        combination[i] = index % 2; // Determine 0 or 1 based on the index.
        index /= 2; // Move to the next bit.
    }
    return combination;
}

// Array equality check method.
static bool AreArraysEqual(int[][] a, int[][] b)
{
    for (int i = 0; i < a.Length; i++)
    {
        for (int j = 0; j < a[i].Length; j++)
        {
            if (a[i][j] != b[i][j])
            {
                return false;
            }
        }
    }
    return true;
}

// Clone Method.
static int[][] CloneArray(int[][] source)
{
    int[][] clone = new int[source.Length][];
    for (int i = 0; i < source.Length; i++)
    {
        clone[i] = (int[])source[i].Clone();
    }
    return clone;
}

//Method to find 0's.
static List<(int, int)> FindChangeablePixels(int[][] matrix)
{
    List<(int, int)> pixels = new List<(int, int)>();
    int rows = matrix.Length;
    int cols = matrix[0].Length;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (matrix[i][j] == 0 && HasAdjacentOne(matrix, i, j))
            {
                pixels.Add((i, j));
            }
        }
    }

    return pixels;
}

//Method to check if pixels have a 1 next to them.
static bool HasAdjacentOne(int[][] matrix, int row, int col)
{
    int rows = matrix.Length;
    int cols = matrix[0].Length;

    //Check neighbors.
    int[] neighbourRow = { -1, 1, 0, 0 };
    int[] neighbourCol = { 0, 0, -1, 1 };

    for (int k = 0; k < 4; k++)
    {
        int newRow = row + neighbourRow[k];
        int newCol = col + neighbourCol[k];

        if (newRow >= 0 && newRow < rows && newCol >= 0 && newCol < cols && matrix[newRow][newCol] == 1)
        {
            return true;
        }
    }

    return false;
}

//In TrainingOne we had 5 int[] arrays to create variations easier. But we need only 1 int[] array which has 25 elements to make neuron operations
//So changing the data type to what we need.
static int[][] ChangeDataTypeOfTrainingData(List<int[][]> oldArray)
{
    List<int[]> newArray = new List<int[]>();

    foreach (var matrix in oldArray)
    {
        //Generate the one dimension array.
        List<int> oneDimensionRow = new List<int>();
        foreach (var row in matrix)
        {
            //Add every element in the row.
            oneDimensionRow.AddRange(row);
        }

        //Add as a 25 element array
        newArray.Add(oneDimensionRow.ToArray());
    }

    return newArray.ToArray();
}


// Create neuron class to take inputs, to create and keep weights and calculate output for each neuron.
public class Neuron
{
    public int[] Inputs { get; set; } // Inputs property.
    public double[] Weights { get; set; } // Weights property.

    private static Random randomWeight = new Random(); // Create a Random object to create random weights.

    // Constructor for neuron. Only takes input count.
    public Neuron(int inputCount)
    {
        // Creating weight and input arrays with the size of input counts.
        Weights = new double[inputCount];
        Inputs = new int[inputCount];

        // Update every element with a random value.
        for (int i = 0; i < inputCount; i++)
        {
            Weights[i] = randomWeight.NextDouble() * (1 - (-1)) - 1; // By last part we ensure that the random range is (-1,1).
        }
    }

    // Calculate output for neuron.
    public double CalculateOutput()
    {
        double sum = 0;

        // Multiply each input with it's weight and add it to sum variable.
        for (int i = 0; i < Inputs.Length; i++)
        {
            sum += Inputs[i] * Weights[i];
        }

        return sum;
    }
}

// Neural network class for neurons.
public class NeuralNetwork
{
    private Neuron[] _neurons;

    // Create neurons that we need.
    public NeuralNetwork(int outputCount, int inputCount)
    {
        _neurons = new Neuron[outputCount];

        for (int i = 0; i < outputCount; i++)
        {
            _neurons[i] = new Neuron(inputCount); //Creating N1, N2 and keeping them in array.
        }
    }

    // Method for updating weights of a neuron.
    public void UpdateWeights(int expectedResult, double trainingRate, int[] currentData)
    {
        // We keep neurons in outputs array.
        double[] outputs = new double[_neurons.Length];

        // For method to update inputs and calculating the outputs. 
        for (int i = 0; i < _neurons.Length; i++)
        {
            _neurons[i].Inputs = currentData;
            double output = _neurons[i].CalculateOutput();
            outputs[i] = output;
        }

        // Condition to check if the given data and the expected result are the same. This is for updating the neuron's weight.
        if (outputs[0] > outputs[1] && expectedResult == 2)
        {
            // Lower the N1 (first neuron) weight. Increase the N2 (second neuron) weight that was being processed.
            for (int i = 0; i < _neurons[0].Weights.Length; i++)
            {
                _neurons[0].Weights[i] -= trainingRate * _neurons[0].Inputs[i];
                _neurons[1].Weights[i] += trainingRate * _neurons[1].Inputs[i];
            }
        }
        else if (outputs[0] < outputs[1] && expectedResult == 1)
        {
            // Lower the N2 (second neuron) weight. Increase the N1 (first neuron) weight that was being processed.
            for (int i = 0; i < _neurons[1].Weights.Length; i++)
            {
                _neurons[1].Weights[i] -= trainingRate * _neurons[1].Inputs[i];
                _neurons[0].Weights[i] += trainingRate * _neurons[0].Inputs[i];
            }
        }
    }

    // Train method for neuron to make it recognize the images as expected.
    public void Train(double trainingRate, int epochs, List<int[][]> trainingSet)
    {
        // Train the neuron for the specified number of epochs.
        for (int i = 0; i < epochs; i++)
        {
            int expectedResult; // Variable for storing expected result.

            // Iterate through every 1 and 2 sets.
            foreach (var set in trainingSet)
            {
                // Iterate through every matrix in 1 and 2 sets.
                for (int j = 0; j < set.Length; j++)
                {
                    // Condition to find the expected result by the position of current matrix in the training set.
                    if (trainingSet.IndexOf(set) == 0)
                    {
                        expectedResult = 1;
                    }
                    else
                    {
                        expectedResult = 2;
                    }

                    // Update weights and train neural network.
                    UpdateWeights(expectedResult, trainingRate, set[j]);
                }
            }
        }
    }

    // Method for showing the final results.
    public void ShowFinalResults(List<int[][]> trainingSet)
    {
        // Variables to hold count of correctly predicted images.
        double correctGuessCount = 0;

        // Array to hold the output values of the neurons for each input.
        double[] outputs = new double[trainingSet.Count];

        // Iterate through each set in the training set.
        foreach (var set in trainingSet)
        {
            // Iterate through each matrix in the current set.
            for (int j = 0; j < set.Length; j++)
            {
                int expectedResult, foundResult;

                // Assign the current input matrix to the neuron's input and calculate the output for each neuron for the current input.
                for (int i = 0; i < _neurons.Length; i++)
                {
                    _neurons[i].Inputs = set[j];
                    double output = _neurons[i].CalculateOutput();
                    outputs[i] = output;
                }

                // Determine the found result based on the neuron's outputs. 
                if (outputs[0] > outputs[1])
                {
                    foundResult = 1;
                }
                else
                {
                    foundResult = 2;
                }

                // Determine the expected result based on the set's position in the training set.
                if (trainingSet.IndexOf(set) == 0)
                {
                    expectedResult = 1;
                }
                else
                {
                    expectedResult = 2;
                }

                // Check if the predicted result matches the expected result.
                // If they match, increment the correctGuess counter.
                if (expectedResult == foundResult)
                {
                    correctGuessCount++;
                }

                // Print the expected and the found result.
                Console.WriteLine($"Beklenen değer: {expectedResult}\nBulunan değer: {foundResult}\n");
            }
        }

        // Calculate the accuracy of neural network's predictions.
        Console.WriteLine("Accuracy is: %" + (correctGuessCount / (trainingSet[0].Length + trainingSet[1].Length) * 100));
    }
}