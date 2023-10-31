/*
 * C implementation of the first example shown
 * in commonlounge.com ML tutorial.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

static inline float randf()
{
	return (float) rand() / (float) RAND_MAX;
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
	srand(time(0));

	float b = randf();
	float w1 = randf();
	
	float it = 1000 * 1000;
	float n = 0.001;

	for (size_t i = 0; i < it; ++i) {
		for (size_t j = 0; j < data_len; ++j) {
			float x = dataset[j][0];
			float y = dataset[j][1];

			printf("%lu\t w1 = %f, b = %f, c = %f\n", i, w1, b, cost_function(w1, b));

			float dw = (predict(x, w1, b) - y) * x;
			float db = (predict(x, w1, b) - y);

			/* printf("dw = %f, db = %f\n", dw, db); */

			b -= n * db;
			w1 -= n * dw;
		}
	}

	/* for (size_t i = 0; i < data_len; ++i) { */
	/* 	float x = dataset[i][0]; */
	/*  */
	/* 	printf("%lu\tpredict(%f) = %f\n", i, x, predict(x, w1, b)); */
	/* } */
}




