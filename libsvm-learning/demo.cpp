#include <iostream>
#include <svm.h>
#include <ctime>
#include <stdlib.h>

int main()
{
	//build svm_parameter
	svm_parameter param;
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	//param.degree = 3;
	param.gamma = 0.5;
	//param.coef0 = 0;
	//param.nu = 0.5;
	param.cache_size = 100;
	param.C = 0.5;
	param.eps = 1e-3;
	//param.p = 0.1;
	param.shrinking = 0;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;

	// build problem
	svm_problem prob;
	int dataSize = 10000;
	prob.l = dataSize;
	prob.y = new double[prob.l];
	prob.x = new svm_node *[prob.l];
	//build data
	srand(time(0));
	double radius2 = 80.0 * 80.0;
	int posNum=0, negNum = 0;
	for(int i = 0; i < dataSize; ++i){
		prob.x[i] = new svm_node[3];
		prob.x[i][0].index = 1;
		prob.x[i][0].value = double(rand()%200 - 100);
		prob.x[i][1].index = 2;
		prob.x[i][1].value = double(rand()%200 - 100);
		prob.x[i][2].index = -1;
		prob.x[i][2].value = 0;
		prob.y[i] = (prob.x[i][0].value * prob.x[i][0].value 
			+ prob.x[i][1].value * prob.x[i][1].value) > radius2 ? -1.0 : 1.0;
		if (prob.y[i] > 0)
		{
			++posNum;
		}else ++negNum;
		if(i%500 == 0) std::cout << "x: "<<prob.x[i][0].value 
			<< "  y: " << prob.x[i][1].value << " preclass : " << prob.y[i] << std::endl;
	}
	std::cout << "pos: " << posNum << "  neg: " << negNum << std::endl;
	//check
	const char* err = svm_check_parameter(&prob, &param);
	if(NULL != err){
		std::cout << errno << std::endl;
		return -1;
	}
	//build model and train
	svm_model *model = svm_train(&prob, &param);
	//predict
	for(int i = 0; i < 20; ++i){
		int n = std::max(0, rand() % dataSize - 1);
		std:: cout << "x : " <<prob.x[i][0].value << "  y: " << prob.x[i][1].value
		<< " class : " << svm_predict(model, prob.x[i]) << " true class " << prob.y[i] << std::endl;
	}
	svm_node x[3];
	x[0].index = 1;
	x[1].index = 2;
	x[2].index = -1;
	//std::cout << "input x : " << std::endl;
	//std::cin >> x[0].value;
	//std::cout << "input y : " << std::endl;
	//std::cin >> x[1].value;
	x[0].value = 0;
	x[1].value = 0;
	x[2].value = -1;

	double d = svm_predict(model, x);
	std::cout << "class is : " << d << std::endl;

	svm_free_and_destroy_model(&model);
	for (int i = 0; i < dataSize; ++i){
		delete[] prob.x[i];
	}
	delete[] prob.x;
	delete[] prob.y;

	return 0;
}
