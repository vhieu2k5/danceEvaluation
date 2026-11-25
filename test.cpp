#include <string>
#include <vector>

/* Model data */
static const struct CatboostModel {
    CatboostModel() = default;
    unsigned int FloatFeatureCount = 3;
    unsigned int CatFeatureCount = 0;
    unsigned int BinaryFeatureCount = 26;
    unsigned int TreeCount = 10;
    std::vector<std::vector<float>> FloatFeatureBorders = {
        {-1.05011964, -0.727960587, -0.274883211, 0.233066961, 0.26163274, 1.68123555},
        {-2.55865455, -1.76195669, -1.19468307, -0.243533954, 0.421992213, 0.66674614, 0.990043759, 1.40895295},
        {-1.59466171, -0.718886733, -0.587888002, -0.442838669, -0.0127533991, 0.378181279, 0.515902877, 0.56651032, 1.09855199, 1.23569095, 1.50451088, 1.67786169}
    };
    unsigned int TreeDepth[10] = {3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
    unsigned int TreeSplits[30] = {22, 15, 17, 22, 16, 20, 4, 23, 5, 8, 16, 6, 10, 1, 19, 25, 3, 12, 14, 21, 24, 2, 7, 18, 11, 0, 9, 20, 13, 18};
    unsigned int BorderCounts[3] = {6, 8, 12};
    float Borders[26] = {-1.05011964f, -0.727960587f, -0.274883211f, 0.233066961f, 0.26163274f, 1.68123555f, -2.55865455f, -1.76195669f, -1.19468307f, -0.243533954f, 0.421992213f, 0.66674614f, 0.990043759f, 1.40895295f, -1.59466171f, -0.718886733f, -0.587888002f, -0.442838669f, -0.0127533991f, 0.378181279f, 0.515902877f, 0.56651032f, 1.09855199f, 1.23569095f, 1.50451088f, 1.67786169f};

    /* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */
    double LeafValues[80][1] = {
        {0.05441110093018104}, {0}, {-0.05259739756584168}, {0}, {0}, {0}, {0.09016696725572859}, {-0.1269592355037558},
        {0.07403401937346377}, {0}, {0.004895297102575026}, {0}, {0}, {0}, {0.07758768885263048}, {-0.1076664899765811},
        {0.09221789629344274}, {-0.1469531432316719}, {-0.0810682059683711}, {-0.03577229242764109}, {0}, {0.06390399258159034}, {0}, {0.08407750623225568},
        {0.03604708245802078}, {0}, {0}, {0}, {-0.07068509910668991}, {0.09929260498882592}, {0.05916552736002734}, {-0.02581274050754667},
        {0.05405154379619587}, {-0.09940509924538865}, {0.01829023529129317}, {0.04530192987474181}, {0.0318914224059046}, {0.07734703640184112}, {-0.1055477570657467}, {0.07620473341751902},
        {0.06185353215012743}, {-0.04156672075350914}, {-0.1380684372459097}, {0.08590135995267609}, {0.05133267848453466}, {-0.03517085248477372}, {0.05380442860841864}, {0},
        {0.01146951559395905}, {0.02504700383067873}, {0}, {-0.01457314293784714}, {0}, {0}, {0}, {-0.01700674449140627},
        {0.03281151090947636}, {-0.0678211866004062}, {-0.007448817702785928}, {0.04933829212940481}, {0.03260358706996867}, {-0.07680853115647139}, {-0.05447544722601759}, {0.06720975818193618},
        {-0.02457789855314456}, {0}, {-0.02282504283596118}, {0}, {0.1367659789823611}, {-0.1391140684295595}, {-0.02203886651220703}, {0.07142944645261919},
        {0.01446948841736277}, {0}, {-0.01687689504111891}, {0}, {0.05886679128710744}, {-0.01046503774723565}, {-0.07059971381657301}, {-0.003365451245715099}
    };
    double Scale = 1;
    double Biases[1] = {0};
    unsigned int Dimension = 1;
} CatboostModelStatic;

/* Model applicator */
std::vector<double> ApplyCatboostModelMulti(
    const std::vector<float>& floatFeatures
) {
    const struct CatboostModel& model = CatboostModelStatic;

    /* Binarize features */
    std::vector<unsigned char> binaryFeatures(model.BinaryFeatureCount);
    unsigned int binFeatureIndex = 0;
    for (unsigned int i = 0; i < model.FloatFeatureCount; ++i) {
        for(unsigned int j = 0; j < model.BorderCounts[i]; ++j) {
            binaryFeatures[binFeatureIndex] = (unsigned char)(floatFeatures[i] > model.Borders[binFeatureIndex]);
            ++binFeatureIndex;
        }
    }

    /* Extract and sum values from trees */
    std::vector<double> results(model.Dimension, 0.0);
    const unsigned int* treeSplitsPtr = model.TreeSplits;
    const auto* leafValuesForCurrentTreePtr = model.LeafValues;
    for (unsigned int treeId = 0; treeId < model.TreeCount; ++treeId) {
        const unsigned int currentTreeDepth = model.TreeDepth[treeId];
        unsigned int index = 0;
        for (unsigned int depth = 0; depth < currentTreeDepth; ++depth) {
            index |= (binaryFeatures[treeSplitsPtr[depth]] << depth);
        }

        for (unsigned int resultIndex = 0; resultIndex < model.Dimension; resultIndex++) {
            results[resultIndex] += leafValuesForCurrentTreePtr[index][resultIndex];
        }

        treeSplitsPtr += currentTreeDepth;
        leafValuesForCurrentTreePtr += 1 << currentTreeDepth;
    }

    std::vector<double> finalResults(model.Dimension);
    for (unsigned int resultId = 0; resultId < model.Dimension; resultId++) {
        finalResults[resultId] = model.Scale * results[resultId] + model.Biases[resultId];
    }
    return finalResults;
}

double ApplyCatboostModel(
    const std::vector<float>& floatFeatures
) {
    return ApplyCatboostModelMulti(floatFeatures)[0];
}

// Also emit the API with catFeatures, for uniformity
std::vector<double> ApplyCatboostModelMulti(
    const std::vector<float>& floatFeatures,
    const std::vector<std::string>&
) {
    return ApplyCatboostModelMulti(floatFeatures);
}

double ApplyCatboostModel(
    const std::vector<float>& floatFeatures,
    const std::vector<std::string>&
) {
    return ApplyCatboostModelMulti(floatFeatures)[0];
}
