/*
 * C implementation of the first example shown
 * in commonlounge.com ML tutorial.
 */

#include <stdio.h>

static const float dataset[][2] = {
	{ 6.65f, 30.7f },
	{ 8.19f, 38.1f },
	{ 8.92f, 44.7f },
	{ 6.21f, 34.9f },
	{ 7.16f, 41.0f },
	{ 5.79f, 33.1f },
	{ 9.17f, 41.4f },
	{ 8.75f, 43.9f },
	{ 6.77f, 31.5f },
	{ 5.65f, 34.3f },
	{ 7.22f, 37.5f },
	{ 7.74f, 39.9f },
	{ 6.58f, 39.2f },
	{ 8.54f, 45.0f },
	{ 5.65f, 29.5f },
	{ 6.49f, 37.5f },
	{ 5.08f, 34.2f },
	{ 8.62f, 42.7f },
	{ 8.47f, 39.2f },
	{ 5.16f, 33.0f }
};

static const size_t data_len = sizeof(dataset) / sizeof(dataset[0]);

/* Predict y given input and model parameters */
static inline float predict(float x1, float w1, float b)
{
	return w1 * x1 + b;
}

/* Calculate cost given model parameters,
 * make predictions for each data (x, ytrue) and
 * calculate squared error */
static float cost_function(float w1, float b)
{
	float err = 0;

	for (size_t i = 0; i < data_len; ++i) {
		float x = dataset[i][0];
		float ytrue = dataset[i][1];
		float ypred = predict(x, w1, b);
		float d = ypred - ytrue;
		err += d * d;
	}

	return err / data_len;
}

int main(void)
{
	/* Bias and weight (try out various values!) */
	float b = 15.f;
	float w1 = 3.f;
	
	float cost = cost_function(w1, b);

	printf("Value of cost function = %f\n", cost);

	for (size_t i = 0; i < data_len; ++i) {
		float x = dataset[i][0];

		printf("%lu\tpredict(%f) = %f\n", i, x, predict(x, w1, b));
	}
}




