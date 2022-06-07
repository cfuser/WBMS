#pragma once
#include <math.h>

double dist(std::vector<double> &x, std::vector<double> &y)
{
	double sum = 0;
	int len = x.size();
	for (int i = 0; i < len; i++)
		sum += (x[i] - y[i]) * (x[i] - y[i]);

	return sqrt(sum);
}


// extern std::vector<int> fa;

int get_Father(std::vector<int> &fa, int u)
{
	if (fa[u] != u)
	{
		fa[u] = get_Father(fa, fa[u]);
	}

	return fa[u];
}

void Union(std::vector<int> &fa, int u, int v)
{
	int fa_u = get_Father(fa, u), fa_v = get_Father(fa, v);
	fa[fa_u] = fa_v;
}
