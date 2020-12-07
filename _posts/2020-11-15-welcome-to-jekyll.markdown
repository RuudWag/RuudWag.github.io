---
layout: post
title:  "How I found a best known solution for VRP!"
date:   2020-11-15 13:39:25 -0600
categories: jekyll update
---
Welcome to my first blog! I have been thinking about writing this blog for a long time. On 18 March 2016, when writing my master thesis at Tilburg University, I achieved the best-known solution for the c1_10_8 of the Gehring & Homberger vehicle routing problems (https://web.archive.org/web/20160320064048/https://www.sintef.no/projectweb/top/vrptw/homberger-benchmark/1000-customers/). For a master student in Operations Research this is one of the biggest achievements you can get. As you can see, I had to use a link from the internet archive because a couple months after my solution someone else found an even better solution. Today in this blog I want to show you what kind of algorithms I have used and explain the awful code I have written.

What is the Vehicle routing problem?

You can download the Gehring & Homberger vehicle routing problems from this website: https://www.sintef.no/projectweb/top/vrptw/homberger-benchmark/1000-customers/

If you open one of the files, the first lines look as follows:

    c1_10_8

    VEHICLE
    NUMBER     CAPACITY
     250          200

    CUSTOMER
    CUST NO.  XCOORD.    YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE TIME
 
    0     250        250          0          0       1824          0
    1     387        297         10        144        390         90
    2       5        297         10        857       1116         90
    3     355        177         20        141        298         90

The vehicle routing problem is pretty easy to understand. In the c1_10_8 case you have a total of 1000 customers. Each customer has a x-coordinate and a y-coordinate. To calculate the distance between two customers you use the euclidean distance. Furthermore each customer has a demand, ready time, due date and service time. You also have the number of vehicles available and the capacity of a given vehicle. The first goal of this problem is to minimize the number of vehicles and the secondary goal is to minimize the total distance traveled. 

The problem described here is called the VRPTW (Vehicle Routing Problem with Time Windows). In the literature you also have the CVRP, DCVRP, VRPPD, VRPDTW, VRPBTW, VRPB and probably many more. See the Wiki page (https://en.wikipedia.org/wiki/Vehicle_routing_problem) for more info.

What makes it so interesting?

The vehicle routing problem is a NP-Hard problem. This means that the time to find the optimal solution for this problem increases exponential with the size of the problem. For smaller problems you can find the optimal solution in reasonable time, but for 1000 customers it would probably take years to solve. To be able to solve those problems within reasonable time you use heuristics. A heuristic is a combination of practical methods to find a solution. An example would be that you start with an empty solution and insert each customer one by one until all customers are in a vehicle. You have no idea whether the solution found is the optimal solution. The Gehring and Homberger instances were created in 2000, but even this year better heuristics were found for some of the instances. 

My research

In September 2015, the time had come. I finished all my courses, and to get my master’s degree I had to write my master thesis. I have a few passions in my life; programming, the field of Operations Research and gaming. At that time I had a crappy laptop and no money to buy a new one. So, I came up with a master plan to write my master thesis about the vehicle routing problem. Instead of using the CPU to solve it I wanted to use the GPU. My professors accepted my proposal and luckily my parents where also convinced to invest in me and give me a brand-new computer with a Nvidia GTX 970 (self build ofcourse). 

I had my new computer, an idea and GTA V. Although I said I like programming, I did not have much experience with it. After some research I figured out that using CUDA with c++ would be the best choice. I never used either of them, so that was going to be fun. Before continue reading, I must warn you. The code you are going to experience in this blog might hurt your eyes. But hey, in the end it worked and I was able to get my best solution.

The algorithm

This research was not about finding a new algorithm but trying to parallelize an existing algorithm to the GPU. I went to the sintef.no website and looked at the papers with the best-known solutions. Most of my algorithm is based on this paper: A powerful route minimization heuristic for the vehicle routing problem with
time windows by Yuichi Nagata and Olli Bräysy  (https://www.sciencedirect.com/science/article/abs/pii/S0167637709000662). Credits to them for figuring out this beautiful algorithm.

Note that all my code can be found here ???. 

To explain the algorithm, I clarify only a few parts of the code. I do not explain everything in detail, because that would be boring.

<details><summary><div style="width:200px;height:25px;border:1px solid #999;">Click to show code!</div></summary>
<p>
{% highlight cpp %}
int main()
{
	int Rnumber;
	ifstream myfile( location );
	if ( !myfile.is_open() )
	{
		cout << "error" << endl;
		return 0;
	}
	int PermArr[ NumberCustomers ];
	int Temp;
	int* h_Route;
	int* d_Route;
	int* h_PrevRoute;
	int* d_PrevRoute;
	int* h_Quantity;
	int CustCorx[ NumberCustomers + 1 ];
	int CustCory[ NumberCustomers + 1 ];
	int* h_CustDem;
	float* h_CustRed;
	float* h_CustDue;
	float* h_CustSer;
	float* h_CustTW;
	int* h_CustCar;
	int* d_EP;
	int* d_EP2;
	int* d_CustDem;
	float* d_CustRed;
	float* d_CustDue;
	float* d_CustSer;
	bool Firstline = false;
	float* h_CustDist;
	float* d_CustDist;
	float* d_Depot;
	float* d_CustTW;
	int* d_CustCar;
	int* h_CarLoad;
	int* d_CarLoad;
	float* h_SlackL;
	float* d_SlackL;
	int* d_PSum;
	int* h_PSum;
	int* d_PSum2;
	int* h_PSum2;
	int* h_ClosestCustomer;
	int* d_ClosestCustomer;
	float* h_CustCost;
	float* d_CustCost;
	int* h_NCars;
	int* d_NCars;
	int* d_j;
	int* d_SEP;
	int* h_SEP;
	int* h_j;
	int* d_NCars2;
	int* d_j2;
	int* d_SEP2;
	float* d_Totaldistance;
	int* d_Insertiterations;
	int* d_Optimizeiterations;
	int* h_Insertiterations;
	int* h_Optimizeiterations;
	gpuErrchk( cudaMallocHost( &h_Insertiterations, sizeof( int ) ) );
	gpuErrchk( cudaMallocHost( &h_Optimizeiterations, sizeof( int ) ) );
	gpuErrchk( cudaMalloc( &d_Insertiterations, sizeof( int ) ) );
	gpuErrchk( cudaMalloc( &d_Optimizeiterations, sizeof( int ) ) );
	gpuErrchk( cudaMalloc( &d_SEP2, sizeof( int ) * Blocks ) );
	gpuErrchk( cudaMalloc( &d_j2, Blocks * sizeof( int ) ) );
	gpuErrchk( cudaMalloc( &d_NCars2, sizeof( int ) * Blocks ) );
	gpuErrchk( cudaMalloc( &d_j, Blocks * sizeof( int ) ) );
	gpuErrchk( cudaMallocHost( &h_j, Blocks * sizeof( int ) ) );
	gpuErrchk( cudaMalloc( &d_Totaldistance, Blocks * sizeof( int ) ) );
	gpuErrchk( cudaMalloc( &d_SEP, sizeof( int ) * Blocks ) );
	gpuErrchk( cudaMallocHost( &h_SEP, sizeof( int ) * Blocks ) );
	gpuErrchk( cudaMalloc( &d_PSum, NumberCustomers * Blocks * sizeof( int ) ) );
	gpuErrchk( cudaMallocHost( &h_PSum, NumberCustomers * Blocks * sizeof( int ) ) );
	gpuErrchk( cudaMalloc( &d_PSum2, NumberCustomers * Blocks * sizeof( int ) ) );
	gpuErrchk( cudaMallocHost( &h_PSum2, NumberCustomers * Blocks * sizeof( int ) ) );
	gpuErrchk( cudaMalloc( &d_NCars, sizeof( int ) * Blocks ) );
	gpuErrchk( cudaMallocHost( &h_NCars, sizeof( int ) * Blocks ) );
	gpuErrchk( cudaMalloc( &d_SlackL, NumberCustomers * Blocks * sizeof( float ) ) );
	gpuErrchk( cudaMallocHost( &h_ClosestCustomer, NumberCustomers * Threads * sizeof( float ) ) );
	gpuErrchk( cudaMalloc( &d_ClosestCustomer, NumberCustomers * Threads * sizeof( float ) ) );
	gpuErrchk( cudaMallocHost( &h_ClosestCustomer, NumberCustomers * Threads * sizeof( float ) ) );
	gpuErrchk( cudaMalloc( &d_ClosestCustomer, NumberCustomers * Threads * sizeof( float ) ) );
	gpuErrchk( cudaMallocHost( &h_CustCost, NumberCustomers * Threads * sizeof( float ) ) );
	gpuErrchk( cudaMalloc( &d_CustCost, NumberCustomers * Threads * sizeof( float ) ) );
	gpuErrchk( cudaMallocHost( &h_SlackL, NumberCustomers * Blocks * sizeof( float ) ) );
	gpuErrchk( cudaMalloc( &d_Route, NumberCustomers * Blocks * sizeof( int ) * 2 ) );
	gpuErrchk( cudaMallocHost( &h_Route, NumberCustomers * Blocks * sizeof( int ) * 2 ) );
	gpuErrchk( cudaMalloc( &d_PrevRoute, NumberCustomers * Blocks * sizeof( int ) ) );
	gpuErrchk( cudaMallocHost( &h_PrevRoute, NumberCustomers * Blocks * sizeof( int ) ) );
	gpuErrchk( cudaMallocHost( &h_Quantity, NumberCustomers * sizeof( float ) ) );
	gpuErrchk( cudaMallocHost( &h_CustTW, (NumberCustomers)*Blocks * sizeof( int ) ) );
	gpuErrchk( cudaMallocHost( &h_CustDem, ( NumberCustomers ) * sizeof( int ) ) );
	gpuErrchk( cudaMallocHost( &h_CustRed, ( NumberCustomers + 1 ) * sizeof( int ) ) );
	gpuErrchk( cudaMallocHost( &h_CustDue, ( NumberCustomers + 1 ) * sizeof( int ) ) );
	gpuErrchk( cudaMallocHost( &h_CustSer, ( NumberCustomers ) * sizeof( int ) ) );
	gpuErrchk( cudaMallocHost( &h_CustDist, ( NumberCustomers + 1 ) * ( NumberCustomers + 1 ) * sizeof( float ) ) );
	gpuErrchk( cudaMallocHost( &h_CustCar, (NumberCustomers)*Blocks * sizeof( float ) ) );
	gpuErrchk( cudaMalloc( &d_CustCar, (NumberCustomers)*Blocks * sizeof( float ) ) );
	gpuErrchk( cudaMalloc( &d_CustTW, (NumberCustomers)*Blocks * sizeof( int ) ) );
	gpuErrchk( cudaMalloc( &d_CustDem, ( NumberCustomers ) * sizeof( int ) ) );
	gpuErrchk( cudaMalloc( &d_CustRed, ( NumberCustomers + 1 ) * sizeof( int ) ) );
	gpuErrchk( cudaMalloc( &d_CustDue, ( NumberCustomers + 1 ) * sizeof( int ) ) );
	gpuErrchk( cudaMalloc( &d_CustSer, ( NumberCustomers ) * sizeof( int ) ) );
	gpuErrchk( cudaMalloc( &d_CustDist, ( NumberCustomers + 1 ) * ( NumberCustomers + 1 ) * sizeof( float ) ) );
	gpuErrchk( cudaMalloc( &d_EP, 100 * Blocks * sizeof( int ) ) );
	gpuErrchk( cudaMalloc( &d_EP2, 100 * Blocks * sizeof( int ) ) );
	gpuErrchk( cudaMallocHost( &h_CarLoad, (NumberCustomers)*Blocks * sizeof( int ) ) );
	gpuErrchk( cudaMalloc( &d_CarLoad, (NumberCustomers)*Blocks * sizeof( int ) ) );

	int* d_PrevRoute2;
	int* d_CarLoad2;
	int* d_Route2;
	int* d_CustCar2;
	float* d_CustTW2;
	gpuErrchk( cudaMalloc( &d_CustCar2, (NumberCustomers)*Blocks * sizeof( float ) ) );
	gpuErrchk( cudaMalloc( &d_PrevRoute2, NumberCustomers * Blocks * sizeof( int ) ) );
	gpuErrchk( cudaMalloc( &d_CarLoad2, (NumberCustomers)*Blocks * sizeof( int ) ) );
	gpuErrchk( cudaMalloc( &d_Route2, NumberCustomers * Blocks * sizeof( int ) * 2 ) );
	gpuErrchk( cudaMalloc( &d_CustTW2, (NumberCustomers)*Blocks * sizeof( int ) ) );

	int* d_PrevRoute3;
	int* d_CarLoad3;
	int* d_Route3;
	int* d_CustCar3;
	float* d_CustTW3;
	gpuErrchk( cudaMalloc( &d_CustTW3, (NumberCustomers)*Blocks * sizeof( int ) ) );
	gpuErrchk( cudaMalloc( &d_CustCar3, (NumberCustomers)*Blocks * sizeof( float ) ) );
	gpuErrchk( cudaMalloc( &d_PrevRoute3, NumberCustomers * Blocks * sizeof( int ) ) );
	gpuErrchk( cudaMalloc( &d_CarLoad3, (NumberCustomers)*Blocks * sizeof( int ) ) );
	gpuErrchk( cudaMalloc( &d_Route3, NumberCustomers * Blocks * sizeof( int ) * 2 ) );

	int* d_PrevRoute4;
	int* d_CarLoad4;
	int* d_Route4;
	int* d_CustCar4;
	float* d_CustTW4;
	gpuErrchk( cudaMalloc( &d_CustTW4, (NumberCustomers)*Blocks * sizeof( int ) ) );
	gpuErrchk( cudaMalloc( &d_CustCar4, (NumberCustomers)*Blocks * sizeof( float ) ) );
	gpuErrchk( cudaMalloc( &d_PrevRoute4, NumberCustomers * Blocks * sizeof( int ) ) );
	gpuErrchk( cudaMalloc( &d_CarLoad4, (NumberCustomers)*Blocks * sizeof( int ) ) );
	gpuErrchk( cudaMalloc( &d_Route4, NumberCustomers * Blocks * sizeof( int ) * 2 ) );
{% endhighlight %}
</p>
</details>

This is the start of the algorithm. Note that some variables have the d_ prefix and others the h_. The d_ stands for device, variables allocated on the GPU and the h_ stands for host, variables allocated on the CPU. You must make sure that enough memory is allocated for the variables by using the CudaMallocHost for the host variables and CudaMalloc for the device variables. Most allocations are dependent on the global variable Blocks. Later, I will explain what this variable means 

Why some variables have up to a number 1 to 4 I don't know. 

<details><summary><div style="width:200px;height:25px;border:1px solid #999;">Click to show code!</div></summary>
<p>
{% highlight cpp %}
int loop = 0;
	int a, b, c, d, e, f, g;
	while ( myfile >> a >> b >> c >> d >> e >> f >> g )
	{
		if ( Firstline )
		{
			CustCorx[ a - 1 ] = b;
			CustCory[ a - 1 ] = c;
			h_CustDem[ a - 1 ] = d;
			h_CustRed[ a - 1 ] = e;
			h_CustDue[ a - 1 ] = f;
			h_CustSer[ a - 1 ] = g;
			loop++;
			if ( f + g + sqrtf( ( CustCorx[ NumberCustomers ] - b ) * ( CustCorx[ NumberCustomers ] - b ) + ( CustCory[ NumberCustomers ] - c ) * ( CustCory[ NumberCustomers ] - c ) ) > h_CustDue[ NumberCustomers ] )
			{
				h_CustDue[ a - 1 ] = h_CustDue[ NumberCustomers ] - sqrtf( ( CustCorx[ NumberCustomers ] - b ) * ( CustCorx[ NumberCustomers ] - b ) + ( CustCory[ NumberCustomers ] - c ) * ( CustCory[ NumberCustomers ] - c ) ) - g;
			}
		}
		else
		{
			CustCorx[ NumberCustomers ] = b;
			CustCory[ NumberCustomers ] = c;
			h_CustRed[ NumberCustomers ] = e;
			h_CustDue[ NumberCustomers ] = f;
			Firstline = true;
		}
	}

	//cout << h_CustRed[NumberCustomers] << " " << h_CustDue[NumberCustomers] << h_CustRed[NumberCustomers-1] << " " << h_CustDue[NumberCustomers-1] << endl;
	for ( int i = 0; i <= NumberCustomers; i++ )
	{
		for ( int j = 0; j <= NumberCustomers; j++ )
		{
			h_CustDist[ i * ( NumberCustomers + 1 ) + j ] = sqrtf( ( CustCorx[ i ] - CustCorx[ j ] ) * ( CustCorx[ i ] - CustCorx[ j ] ) + ( CustCory[ i ] - CustCory[ j ] ) * ( CustCory[ i ] - CustCory[ j ] ) );
		}
	}

	pair<float, int> SortedCust[ N ];

	/**find the amount of threads closest customers**/

	for ( int i = 0; i < NumberCustomers; i++ )
	{
		for ( int j = 0; j < N; j++ )
		{
			//float Pen;
			//Pen = h_CustRed[j] + h_CustDist[i*(N + 1) + j];
			/*if (Pen > h_CustDue[i])
			{
				Pen = 1000;
			}*/
			/*else if (Pen < h_CustRed[i])
			{
				Pen = (h_CustRed[i] - Pen) * 0.1;
			}*/
			/*else
			{
				Pen = 0;
			}*/

			SortedCust[ j ] = make_pair( h_CustDist[ i * ( N + 1 ) + j ], j );

		}
		sort( begin( SortedCust ), end( SortedCust ) );

		for ( int j = 0; j < Threads; j++ )
		{
			h_ClosestCustomer[ i * Threads + j ] = SortedCust[ j + 1 ].second;
			h_CustCost[ i * Threads + j ] = SortedCust[ j + 1 ].first;
		}
		sort( &h_ClosestCustomer[ i * Threads ], &h_ClosestCustomer[ i * Threads + Threads ] );
	}


	/************************************/
	/*inalatization of random customers*/
	/**********************************/
	for ( int k = 0; k < NumberCustomers; k++ )
	{
		PermArr[ k ] = k;
	}
	for ( int j = 0; j < Blocks; j++ )
	{
		h_NCars[ j ] = N;
		h_j[ j ] = 0;
		for ( int i = 0; i < NumberCustomers; i++ )
		{
			h_PSum[ i + NumberCustomers * j ] = 1;
			Rnumber = rand() % ( NumberCustomers - i );
			if ( Rnumber == ( NumberCustomers - i - 1 ) )
			{
				h_Route[ i + 2 * j * NumberCustomers + NumberCustomers ] = PermArr[ Rnumber ];
				h_Route[ i + 2 * j * NumberCustomers ] = NumberCustomers;
				h_CustTW[ i + j * NumberCustomers ] = fmaxf( h_CustDist[ i * ( NumberCustomers + 1 ) + NumberCustomers ], h_CustRed[ i ] );
				h_SlackL[ i + j * NumberCustomers ] = h_CustDue[ i ] - h_CustTW[ i + j * NumberCustomers ];
				if ( h_CustDue[ NumberCustomers ] - ( h_CustTW[ i + j * NumberCustomers ] + 10 + h_CustDist[ i * ( NumberCustomers + 1 ) + NumberCustomers ] ) < h_SlackL[ i + j * NumberCustomers ] )
				{
					h_SlackL[ i + j * NumberCustomers ] = h_CustDue[ NumberCustomers ] - ( h_CustTW[ i + j * NumberCustomers ] + 10 + h_CustDist[ i * ( NumberCustomers + 1 ) + NumberCustomers ] );
				}
				h_PrevRoute[ PermArr[ Rnumber ] + j * NumberCustomers ] = i + NumberCustomers;
				h_CustCar[ PermArr[ Rnumber ] + j * NumberCustomers ] = i;
				h_CarLoad[ i + j * NumberCustomers ] = h_CustDem[ PermArr[ Rnumber ] ];

			}
			else
			{
				h_Route[ i + 2 * j * NumberCustomers + NumberCustomers ] = PermArr[ Rnumber ];
				h_Route[ i + 2 * j * NumberCustomers ] = NumberCustomers;
				h_CustTW[ i + j * NumberCustomers ] = fmaxf( h_CustDist[ i * ( NumberCustomers + 1 ) + NumberCustomers ], h_CustRed[ i ] );
				h_SlackL[ i + j * NumberCustomers ] = h_CustDue[ i ] - h_CustTW[ i + j * NumberCustomers ];
				if ( h_CustDue[ NumberCustomers ] - ( h_CustTW[ i + j * NumberCustomers ] + 10 + h_CustDist[ i * ( NumberCustomers + 1 ) + NumberCustomers ] ) < h_SlackL[ i + j * NumberCustomers ] )
				{
					h_SlackL[ i + j * NumberCustomers ] = h_CustDue[ NumberCustomers ] - ( h_CustTW[ i + j * NumberCustomers ] + 10 + h_CustDist[ i * ( NumberCustomers + 1 ) + NumberCustomers ] );
				}
				h_PrevRoute[ PermArr[ Rnumber ] + j * NumberCustomers ] = i + NumberCustomers;
				h_CustCar[ PermArr[ Rnumber ] + j * NumberCustomers ] = i;
				h_CarLoad[ i + j * NumberCustomers ] = h_CustDem[ PermArr[ Rnumber ] ];

				Temp = PermArr[ Rnumber ];
				PermArr[ Rnumber ] = PermArr[ NumberCustomers - i - 1 ];
				PermArr[ NumberCustomers - i - 1 ] = Temp;


			}

		}
	}

	/***************/
	/*PMA ALGORITHM*/
	/***************/
	curandState* devStates;
	gpuErrchk( cudaMalloc( (void**)&devStates, Threads * Blocks * sizeof( curandState ) ) );
	setup_kernel << <Blocks, Threads >> > ( devStates );
	gpuErrchk( cudaMemcpy( d_j, h_j, Blocks * sizeof( int ), cudaMemcpyHostToDevice ) );
	gpuErrchk( cudaMemcpy( d_NCars, h_NCars, Blocks * sizeof( int ), cudaMemcpyHostToDevice ) );
	gpuErrchk( cudaMemcpy( d_PSum, h_PSum, NumberCustomers * Blocks * sizeof( int ), cudaMemcpyHostToDevice ) );
	gpuErrchk( cudaMemcpy( d_ClosestCustomer, h_ClosestCustomer, NumberCustomers * Threads * sizeof( int ), cudaMemcpyHostToDevice ) );
	gpuErrchk( cudaMemcpy( d_CustCost, h_CustCost, NumberCustomers * Threads * sizeof( int ), cudaMemcpyHostToDevice ) );
	gpuErrchk( cudaMemcpy( d_Route, h_Route, NumberCustomers * Blocks * sizeof( int ) * 2, cudaMemcpyHostToDevice ) );
	gpuErrchk( cudaMemcpy( d_PrevRoute, h_PrevRoute, NumberCustomers * Blocks * sizeof( int ), cudaMemcpyHostToDevice ) );
	gpuErrchk( cudaMemcpy( d_CustCar, h_CustCar, NumberCustomers * Blocks * sizeof( int ), cudaMemcpyHostToDevice ) );
	gpuErrchk( cudaMemcpy( d_CustTW, h_CustTW, NumberCustomers * Blocks * sizeof( int ), cudaMemcpyHostToDevice ) );
	gpuErrchk( cudaMemcpy( d_CustDem, h_CustDem, NumberCustomers * sizeof( int ), cudaMemcpyHostToDevice ) );
	gpuErrchk( cudaMemcpy( d_CustRed, h_CustRed, ( NumberCustomers + 1 ) * sizeof( int ), cudaMemcpyHostToDevice ) );
	gpuErrchk( cudaMemcpy( d_CustDue, h_CustDue, ( NumberCustomers + 1 ) * sizeof( int ), cudaMemcpyHostToDevice ) );
	gpuErrchk( cudaMemcpy( d_CustDist, h_CustDist, ( NumberCustomers + 1 ) * ( NumberCustomers + 1 ) * sizeof( float ), cudaMemcpyHostToDevice ) );
	gpuErrchk( cudaMemcpy( d_CarLoad, h_CarLoad, ( NumberCustomers ) * sizeof( int ) * Blocks, cudaMemcpyHostToDevice ) );
	printf( "GOOO" );
{% endhighlight %}
</p>
</details>

This code mainly responsible for reading the data files and copying the values to the GPU by using the cudaMemcpy. This function copies data from the host (CPU) to the Device (GPU) when using cudaMemcpyHostToDevice or vice versa by using cudaMemcpyDeviceToHost. 

In the code above I create the data structures which are used throughout the whole algorithm. The easy ones are the arrays like CustDem, CustRed, CustDue and so on. CustDem[0] will give you the demand for customer 0. The most important data structure is the Route[] array.  Route[0] has the value to which customer you are going from customer 0. At the start of the algorithm every customer has its own vehicle. To be able to model this I made the size of the Route array N*2. The first N entries are for the customers and the second N entries are for the vehicles. A route starts at a vehicle, goes to the customers and ends at the same vehicle. For example, assume that you have a problem with 3 customers. Then the indices 0, 1 and 2 are for the customers. The vehicles have indices 3, 4 and 5. An example route could be going from vehicle 3 to customer 0 to customer 2 back to vehicle 3. In that case the values for Route will be:  Route[3]=0,  Route[0]=2 and Route[2] = 3.
To loop over a route I use the following code often:

{% highlight cpp linenos%}
int customer = Route [ 3 ];
while ( customer < N )
{
	customer = Route [ customer ];
}
				} {% endhighlight %}
I loop through the route until the customer has a value higher than N. I also have a data structure to loop in reverse thought the route, PrevRoute[].

The advantage of this structure is that it is easy to insert and remove customers from a route. If I for example want to remove customer 0 from our previous route, I can set Route[3] to 2 and PrevRoute[2] to 3. I set d_Route[0] to -1, such that I know it is not in the current solution. Inserting is also easy, but I leave that as an exercise for the readers.

Another important variable is the h_NCars. This number gives the number of currently active vehicles. In our small example this number will start with 3, so that means that vehicles 3, 4 and 5 are active and  Route[3],  Route[4] and  Route[5] have actual routes. I try to decrease the number of vehicles in the algorithm, so after some time NCars will hopefully be 2. The algorithm will switch vehicles in such a way that that the vehicles up to NCars are valid vehicles. So, if I remove vehicle 3, I will switch the customers from vehicle 5 to vehicle 3 such that vehicle 3 and 4 are valid routes. I must make sure to not use Route[5], because it has an invalid route.

<details><summary><div style="width:200px;height:25px;border:1px solid #999;">Click to show code!</div></summary>
<p>
{% highlight cpp linenos%}
h_Optimizeiterations[ 0 ] = 2000;
	for ( int i = 0; i < Numberround; i++ )
	{
		gpuErrchk( cudaDeviceSynchronize() );
		if ( h_Insertiterations[ 0 ] < 80 )
		{
			h_Insertiterations[ 0 ] = 50;// i * 10 + 25;
		}
		gpuErrchk( cudaMemcpy( d_Insertiterations, h_Insertiterations, sizeof( int ), cudaMemcpyHostToDevice ) );

		if ( !MinCarRdy )
		{
			PMA << <Blocks, Threads >> > ( d_Route, d_PrevRoute, d_CustTW, d_CustCar, d_CustRed, d_CustDue, d_CustDem, d_CustDist, devStates, d_EP, d_CarLoad, d_PSum, d_Route2, d_PrevRoute2, d_CustCar2, d_CarLoad2, d_CustTW2, d_ClosestCustomer, d_CustCost, d_NCars, d_j, d_SEP, d_EP2, d_Route3, d_PrevRoute3, d_CustCar3, d_CarLoad3, d_CustTW3, d_Totaldistance, d_j2, d_SEP2, d_NCars2, d_Insertiterations );
		}
		if ( i % 4 == 0 && i > 0 )
		{
			h_Optimizeiterations[ 0 ] = 3500;
			gpuErrchk( cudaMemcpy( d_Optimizeiterations, h_Optimizeiterations, sizeof( int ), cudaMemcpyHostToDevice ) );
		}
		else
		{
			h_Optimizeiterations[ 0 ] = 1000;
			gpuErrchk( cudaMemcpy( d_Optimizeiterations, h_Optimizeiterations, sizeof( int ), cudaMemcpyHostToDevice ) );
		}
		duration = ( std::clock() - start ) / (double)CLOCKS_PER_SEC;
		cout << " " << endl;
		cout << "printf: " << duration << '\n' << endl;
		//h_Optimizeiterations[0] = 1;
		//gpuErrchk(cudaMemcpy(d_Optimizeiterations, h_Optimizeiterations, sizeof(int), cudaMemcpyHostToDevice));
		gpuErrchk( cudaDeviceSynchronize() );

		//MinCarRdy = true;
		/*for (int i = 0; i < Blocks; i++)
		{
		if (h_NCars[i] > BNCars || h_j[i] != h_SEP[i])
		{
		MinCarRdy = false;
		}

		}*/

		OPT << <Blocks, Threads >> > ( d_Route, d_PrevRoute, d_CustTW, d_CustCar, d_CustRed, d_CustDue, d_CustDem, d_CustDist, devStates, d_EP, d_CarLoad, d_PSum, d_Route2, d_PrevRoute2, d_CustCar2, d_CarLoad2, d_ClosestCustomer, d_CustCost, d_NCars, d_Route3, d_PrevRoute3, d_CustCar3, d_CarLoad3, d_CustTW3, d_Route4, d_PrevRoute4, d_CustCar4, d_CarLoad4, d_CustTW4, d_PSum2, d_j, d_EP2, d_Totaldistance, d_SEP, d_j2, d_SEP2, d_NCars2, d_Optimizeiterations );
	}
{% endhighlight %}
</p>
</details>

On line 13 I am going to call the algorithm on the GPU. As parameters I pass in the device variables created before and I must set the number of blocks and thread I want to use. 

To explain threads and blocks I must go into detail on how the GPU works. A GPU consist of multiple streaming multiprocessors(SMs), the GTX 970 has 13 SMs. A block will be assigned to one of those SMs. Each block will have the number of threads you have assigned. In my code I use 100 blocks with 256 threads. So, in total I will have 100*256 = 25 600 threads running concurrently to each other.

Earlier you saw that a lot of allocations are dependent on those blocks. The reason is that to achieve enough parallelism, I solve the same problem 100 times with different random seeds. Every block represents a different solution and has its own dedicated Route[]. That is why I allocate (NumberCustomers * 2 * Blocks) for this variable. 

<details><summary><div style="width:200px;height:25px;border:1px solid #999;">Click to show code!</div></summary>
<p>
{% highlight cpp linenos%}
__global__ void PMA( int* G_Route, int* PCust, float* TW, int* Car, const float* __restrict__ Red, const float* __restrict__ Due, int* Dem, const float* __restrict__ CustDist, curandState* state, int* EP, int* CarLoad, int* CustPen, int* Custtemp, int* PCusttemp, int* Cartemp, int* CarLoadtemp, float* TWtemp, const int* __restrict__ BestCust, const float* __restrict__ CustCost, int* NumberCars, int* G_j, int* G_SEP, int* EP2, int* Custtemp2, int* PCusttemp2, int* Cartemp2, int* CarLoadtemp2, float* TWtemp2, float* TDist, int* G_j2, int* G_SEP2, int* G_NCars2, const int* Insertiterations )
{

	int idx = threadIdx.x;
	float Ser = Servicetime;
	int MaxLoad = Totalmaxload;
	int Sh = blockIdx.x * N;
	curandState LocalState = state[ threadIdx.x + blockIdx.x * blockDim.x ];

	__shared__ int NCars;
	//__shared__ int Dem[N];
	//__shared__ float Red[N];
	//__shared__ float Due[N];
	//__shared__ float Dep[N];
	__shared__ int Cust[ N * 2 ];
	//__shared__ int PCust[N];
	//__shared__ float TW[N];
	//__shared__ int Car[N];
	//__shared__ int CarLoad[N];
	//__shared__ int CustPen[N];
	__shared__ int RNumber;
	__shared__ bool found;
	__shared__ int SEP;
	__shared__ int j;
	__shared__ float Best[ Threads ];
	__shared__ int Key[ Threads ];
	__shared__ float Prom[ 128 ];
	__shared__ int KeyProm[ 128 ];
	__shared__ int k;
	__shared__ float Penalty;
	__shared__ int PBest;


	while ( idx < N )
	{

		//Dem[idx] = CustDem[idx];
		//Red[idx] = CustRed[idx];
		//Due[idx] = CustDue[idx];
		Cust[ idx ] = G_Route[ idx + 2 * blockIdx.x * N ];
		Cust[ idx + N ] = G_Route[ idx + 2 * blockIdx.x * N + N ];
		//PCust[idx] = G_PRoute[idx + blockIdx.x*N];
		//TW[idx] = G_TW[idx + blockIdx.x*N];
		//Car[idx] = G_Car[idx + blockIdx.x*N];
		//CarLoad[idx] = G_CarLoad[idx + blockIdx.x*N];
		//CustPen[idx + Sh] = 1;
		idx += blockDim.x;

	}


	NCars = NumberCars[ blockIdx.x ];
	idx = threadIdx.x;
	j = G_j[ blockIdx.x ];
	SEP = G_SEP[ blockIdx.x ];
	__syncthreads();
{% endhighlight %}
</p>
</details>

There are three main types of memory on the GPU, the global memory, shared memory and registers. The global memory is the biggest, in my case it is 4 GB and is the slowest. All the d_ variables are allocated on the global memory. If you access a d_ variable often, you can make it faster by using the shared memory. To allocate on the shared memory I am using the __shared__ keyword. Every streaming processor has 96kb which must be shared by the blocks using that streaming processor.

The global memory is accasible by any thread. The shared memory is only accisible for the threads within the same block. As last you have the registers, this is the local storage for a thread. This is small, but super fast. Only the thread itself can access this memory.

How you structure your code is important on the GPU. A GPU runs multiple threads at the same time, this is called the Same Instruction Multiple Threads (SIMT) model. What this means is that every line of code is executed by 32 threads at the same time (called a warp). Every thread has a number which can be accessed threadIdx.x. For the first thread in a block this will hold value 0, for the second 1 and so on. You also have something similar for the blocks, blockIdx.x. I use those numbers to be able to access the correct route in a block. Using G_Route[threadIdx.x + N*2* blockIdx.x] will access for the first block and the first warp   G_Route[0], G_Route[1] ...,G_Route[31] at the same time. This is perfect because all the values are next to each other in the global memory. If for example I would need G_Route[0] and G_Route[100] in the same warp this would be less efficient on the global memory.

To summarize this, if you read data, you want to make sure make sure it is next to each other.

I wanted to keep the part on the GPU short, but I must mention one more important factor. If you have for some reason the following code: 

{% highlight cpp linenos%}
if (threadIdx.x == 0)
{
    DoSomeExpensiveCalculation();
}
{% endhighlight %}

The warp will only use one thread to execute DoSomeExpensiveCalculation(). The other 31 threads must wait for for this thread to finish. This warp will only use 1/32 of its potential power.

Going back to the code, you will see that I use __syncthreads(); often. This is a barrier for the threads within a block. All threads must reach this point before any thread within the block can continue.

In this part I  introduce a couple of important variables EP, SEP and CEP and j. EP= Ejection pool, SEP = Size ejection pool, CEP = current customer from ejection pool and j = current position within the ejection pool. Let me explain what the ejection pool is by example. When I start the algorithm, every customer is assigned to exactly one vehicle. I start out with the following routes 3 -> 0 -> 3, 4 -> 1 -> 4 and 5 -> 2 -> 5. The main goal of the algorithm is to reduce the number of vehicles used. At the start the EP is empty, so j and SEP are 0. In the following code I am going to remove a vehicle:

<details><summary><div style="width:200px;height:25px;border:1px solid #999;">Click to show code!</div></summary>
<p>
{% highlight cpp linenos%}
if ( j == SEP )
		{
			stop = 0;
			Key[ threadIdx.x ] = threadIdx.x;
			Best[ threadIdx.x ] = 0;
			if ( threadIdx.x < NCars )
			{
				temp = Cust[ threadIdx.x + N ];
				while ( temp < N )
				{
					Best[ threadIdx.x ] += CustPen[ temp + Sh ];
					temp = Cust[ temp ];
				}

			}
			else
			{
				Best[ threadIdx.x ] = 10000.0f;
			}
			__syncthreads();
			idx = threadIdx.x;
			//	if (idx < 512){ if (Best[idx] > Best[idx + 512]){ Best[idx] = Best[idx + 512]; Key[idx] = Key[idx + 512]; } } __syncthreads();
			//if (idx < 256){ if (Best[idx] > Best[idx + 256]){ Best[idx] = Best[idx + 256]; Key[idx] = Key[idx + 256]; } } __syncthreads();
			if ( idx < 128 )
			{
				if ( Best[ idx ] > Best[ idx + 128 ] )
				{
					Best[ idx ] = Best[ idx + 128 ]; Key[ idx ] = Key[ idx + 128 ];
				}
			} __syncthreads();
			if ( idx < 64 )
			{
				if ( Best[ idx ] > Best[ idx + 64 ] )
				{
					Best[ idx ] = Best[ idx + 64 ]; Key[ idx ] = Key[ idx + 64 ];
				}
			} __syncthreads();
			if ( idx < 32 )
			{
				if ( Best[ idx ] > Best[ idx + 32 ] )
				{
					Best[ idx ] = Best[ idx + 32 ]; Key[ idx ] = Key[ idx + 32 ];
				}
				if ( Best[ idx ] > Best[ idx + 16 ] )
				{
					Best[ idx ] = Best[ idx + 16 ]; Key[ idx ] = Key[ idx + 16 ];
				}
				if ( Best[ idx ] > Best[ idx + 8 ] )
				{
					Best[ idx ] = Best[ idx + 8 ]; Key[ idx ] = Key[ idx + 8 ];
				}
				if ( Best[ idx ] > Best[ idx + 4 ] )
				{
					Best[ idx ] = Best[ idx + 4 ]; Key[ idx ] = Key[ idx + 4 ];
				}
				if ( Best[ idx ] > Best[ idx + 2 ] )
				{
					Best[ idx ] = Best[ idx + 2 ]; Key[ idx ] = Key[ idx + 2 ];
				}
				if ( Best[ idx ] > Best[ idx + 1 ] )
				{
					Best[ idx ] = Best[ idx + 1 ]; Key[ idx ] = Key[ idx + 1 ];
				}
			}


			__syncthreads();

			if ( idx == 0 )
			{
				RNumber = Key[ 0 ];
			}
			__syncthreads();
			temp = Cust[ RNumber + N ];
			int Counter = 0;
			while ( temp != N && idx > Counter )
			{

				temp = Cust[ temp ];
				Counter++;
			}
			__syncthreads();
			if ( temp < N )
			{

				EP[ idx + blockIdx.x * 100 ] = temp;
				Cust[ temp ] = -1;
			}
			else
			{
				SEP = Counter;
			}
			if ( idx == 0 )
			{
				NCars--;
				Cust[ RNumber + N ] = Cust[ NCars + N ];
				PCust[ Cust[ RNumber + N ] + Sh ] = RNumber + N;
				CarLoad[ RNumber + Sh ] = CarLoad[ NCars + Sh ];
			}
			__syncthreads();
			while ( idx < N )
			{
				if ( Car[ idx + Sh ] == NCars )
				{
					Car[ idx + Sh ] = RNumber;

				}
				idx = idx + blockDim.x;
			}
			idx = threadIdx.x;
			__syncthreads();
			j = 0;
		}
		else
		{
			if ( !Genetic )
			{
				if ( threadIdx.x < SEP )
				{
					EP[ threadIdx.x + 100 * blockIdx.x ] = EP2[ threadIdx.x + 100 * blockIdx.x ];
				}
				__syncthreads();
			}
		}
{% endhighlight %}
</p>
</details>

I remove the vehicle where the customers combined have the least penalty. I will explain what penalty means later on. If I remove the vehicle with route 4 -> 1 -> 4, customer 1 is put in the EP. I only removed one customer, so the SEP will be 1. the variable j will remain 0, it is pointing to the customer which is longest in the EP.

Deciding which vehicle should be removed is done in parallel. Every thread with a threadIdx.x smaller than NCars will evaluate a vehicle. A thread with a value higher than NCars will do nothing. The threads compare their value with each other to find the best vehicle to remove. 

At line 93 I used a pattern which I told you before you shouldn't do. I used the if(idx==0). This was a necessary evil, because updating the routes of the solution can only be done by one thread at a time. In this part I decrease the NCars by one and make sure that the Route variable for this block is updated accordingly.

I have removed a vehicle from the solution, the next goal is to add it back to the solution without using an extra vehicle.

<details><summary><div style="width:200px;height:25px;border:1px solid #999;">Click to show code!</div></summary>
<p>
{% highlight cpp linenos%}
			found = true;
			Best[ threadIdx.x ] = 10000.0f;
			CEP = EP[ j + blockIdx.x * 100 ];
			//idx = threadIdx.x;
			possible = false;
			//BestPlace = idx;
			/*if (CEP - 226 >= 0)
			{
			idx = CEP - 226 + threadIdx.x;
			}
			else
			{
			idx = threadIdx.x;
			}*/
			idx = threadIdx.x;
			while ( idx < N )
			{
				possible = false;
				if ( Cust[ idx ] >= 0 && CarLoad[ Car[ idx + Sh ] + Sh ] + Dem[ CEP ] <= MaxLoad )
				{
					temp = idx;
					temp2 = Cust[ idx ];
					Arrival = fmaxf( TW[ temp + Sh ] + Ser + CustDist[ CEP * ( N + 1 ) + temp ], Red[ CEP ] );
					if ( Arrival <= Due[ CEP ] )
					{
						possible = true;
						temp = CEP;
						Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
					}
					while ( possible && temp2 < N && Arrival > TW[ temp2 + Sh ] )
					{
						if ( Arrival > Due[ temp2 ] )
						{
							possible = false;
						}
						else
						{
							temp = temp2;
							temp2 = Cust[ temp2 ];
							Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
						}
					}
					if ( possible )
					{
						MDist = ( CustDist[ idx * ( N + 1 ) + CEP ] + CustDist[ CEP * ( N + 1 ) + Cust[ idx ] ] - CustDist[ idx * ( N + 1 ) + Cust[ idx ] ] ) * curand_uniform( &LocalState );
						if ( MDist < Best[ threadIdx.x ] )
						{
							Best[ threadIdx.x ] = MDist;
							BestPlace = idx;
							Key[ threadIdx.x ] = threadIdx.x;

						}
					}
				}
				idx += blockDim.x;
			}
{% endhighlight %}
</p>
</details>

First, I get the customer which has been longest in the EP and call it CEP. The first step of the algorithm is to check if I can insert it in any other vehicle at any place without violation the max capacity of the vehicle or by violation one of the time windows of any of the other customers. If multiple insertions are possible, I pick the one which added the least amount of distance.

In our little example you know that Route[0] = 3, Route [1] = -1 and Route[2] = 5. This is easy to parallelize, because thread 0 can check the insertion of CEP between 0 and 3 and thread 2 between 2 and 5. Thread 1 will be idle because customer 1 is not in a route.

If the insertion phase succeeds, I will try to insert the next customer in the ejection pool. If I do not have any customers left, I will remove another vehicle. In case I could not find a feasible insertion, I go to the next phase, the squeeze phase.

<details><summary><div style="width:200px;height:25px;border:1px solid #999;">Click to show code!</div></summary>
<p>
{% highlight cpp linenos%}
			/********************/
			/**Part 2: Squeeze**/
			/******************/
			if ( !found )
			{
				/***************************************/
				/**Add Customer to least penalty cost**/
				/*************************************/
				idx = threadIdx.x;
				Best[ idx ] = 10000.0f;
				while ( idx < N )
				{
					Custtemp[ idx + 2 * Sh ] = Cust[ idx ];
					Custtemp[ idx + 2 * Sh + N ] = Cust[ idx + N ];
					PCusttemp[ idx + Sh ] = PCust[ idx + Sh ];
					TWtemp[ idx + blockIdx.x * N ] = TW[ idx + Sh ];
					Cartemp[ idx + Sh ] = Car[ idx + Sh ];
					CarLoadtemp[ idx + Sh ] = CarLoad[ idx + Sh ];

					idx += blockDim.x;
				}
				__syncthreads();
				idx = threadIdx.x;
				possible = false;

				float LPenalty;
				float APenalty;
				/*if (CEP - 226 >= 0)
				{
				idx = CEP - 226 + threadIdx.x;
				}
				else
				{
				idx = threadIdx.x;
				}*/
				idx = threadIdx.x;
				while ( idx < N )
				{
					if ( Cust[ idx ] >= 0 )
					{
						LPenalty = 0;
						temp = idx;
						temp2 = Cust[ idx ];
						Arrival = fmaxf( TW[ temp + Sh ] + Ser + CustDist[ CEP * ( N + 1 ) + temp ], Red[ CEP ] );
						LPenalty += fmaxf( Arrival - Due[ CEP ], 0.0f );
						APenalty = fmaxf( 0, Arrival - Due[ CEP ] );
						temp = CEP;
						Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
						APenalty = fmaxf( APenalty, Arrival - Due[ temp2 ] );
						while ( temp2 != N && Arrival != TW[ temp2 + Sh ] )
						{
							APenalty = fmaxf( APenalty, Arrival - Due[ temp2 ] );
							LPenalty += fmaxf( Arrival - Due[ temp2 ], 0.0f );
							temp = temp2;
							temp2 = Cust[ temp2 ];
							Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
						}
						LPenalty += fmaxf( CarLoad[ Car[ idx + Sh ] + Sh ] + Dem[ CEP ] - MaxLoad, 0.0f );
						APenalty += fmaxf( CarLoad[ Car[ idx + Sh ] + Sh ] + Dem[ CEP ] - MaxLoad, 0.0f );
						if ( APenalty < Best[ threadIdx.x ] )
						{
							BMDist = LPenalty;
							Best[ threadIdx.x ] = APenalty;
							BestPlace = idx;
							Key[ threadIdx.x ] = threadIdx.x;
						}
					}
					idx += blockDim.x;
				}
				if ( CEP < 250 )
				{
					idx = N + threadIdx.x;
					while ( idx < N + NCars )
					{
						LPenalty = 0;
						temp = Cust[ idx ];
						Arrival = fmaxf( CustDist[ ( N ) * ( N + 1 ) + CEP ], Red[ CEP ] );
						LPenalty += fmaxf( Arrival - Due[ CEP ], 0.0f );
						APenalty = fmaxf( 0, Arrival - Due[ CEP ] );
						Arrival = fmaxf( Arrival + Ser + CustDist[ CEP * ( N + 1 ) + temp ], Red[ temp ] );
						temp2 = temp;
						APenalty = fmaxf( APenalty, Arrival - Due[ temp2 ] );
						while ( temp2 != N && Arrival != TW[ temp2 + Sh ] )
						{
							APenalty = fmaxf( APenalty, Arrival - Due[ temp2 ] );
							LPenalty += fmaxf( Arrival - Due[ temp2 ], 0.0f );
							temp = temp2;
							temp2 = Cust[ temp2 ];
							Arrival = Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ];

						}
						LPenalty += fmaxf( CarLoad[ idx - N + Sh ] + Dem[ CEP ] - MaxLoad, 0.0f );
						APenalty += fmaxf( CarLoad[ idx - N + Sh ] + Dem[ CEP ] - MaxLoad, 0.0f );
						if ( APenalty < Best[ threadIdx.x ] )
						{
							BMDist = LPenalty;
							Best[ threadIdx.x ] = APenalty;
							BestPlace = idx;
							Key[ threadIdx.x ] = threadIdx.x;

						}
						idx = idx + blockDim.x;
					}
				}
				__syncthreads();
				/***************************/
				/**Sort all the penalties**/
				/*************************/
				idx = threadIdx.x;
				//	if (idx < 512){ if (Best[idx] > Best[idx + 512]){ Best[idx] = Best[idx + 512]; Key[idx] = Key[idx + 512]; } } __syncthreads();
				//if (idx < 256){ if (Best[idx] > Best[idx + 256]){ Best[idx] = Best[idx + 256]; Key[idx] = Key[idx + 256]; } } __syncthreads();
				if ( idx < 128 )
				{
					if ( Best[ idx ] > Best[ idx + 128 ] )
					{
						Best[ idx ] = Best[ idx + 128 ]; Key[ idx ] = Key[ idx + 128 ];
					}
				} __syncthreads();
				if ( idx < 64 )
				{
					if ( Best[ idx ] > Best[ idx + 64 ] )
					{
						Best[ idx ] = Best[ idx + 64 ]; Key[ idx ] = Key[ idx + 64 ];
					}
				} __syncthreads();
				if ( idx < 32 )
				{
					if ( Best[ idx ] > Best[ idx + 32 ] )
					{
						Best[ idx ] = Best[ idx + 32 ]; Key[ idx ] = Key[ idx + 32 ];
					}
					if ( Best[ idx ] > Best[ idx + 16 ] )
					{
						Best[ idx ] = Best[ idx + 16 ]; Key[ idx ] = Key[ idx + 16 ];
					}
					if ( Best[ idx ] > Best[ idx + 8 ] )
					{
						Best[ idx ] = Best[ idx + 8 ]; Key[ idx ] = Key[ idx + 8 ];
					}
					if ( Best[ idx ] > Best[ idx + 4 ] )
					{
						Best[ idx ] = Best[ idx + 4 ]; Key[ idx ] = Key[ idx + 4 ];
					}
					if ( Best[ idx ] > Best[ idx + 2 ] )
					{
						Best[ idx ] = Best[ idx + 2 ]; Key[ idx ] = Key[ idx + 2 ];
					}
					if ( Best[ idx ] > Best[ idx + 1 ] )
					{
						Best[ idx ] = Best[ idx + 1 ]; Key[ idx ] = Key[ idx + 1 ];
					}
				}
				__syncthreads();
				/************************************/
				/***Add Customer to Best location***/
				/**********************************/
				if ( Best[ 0 ] < 10000.0f )
				{
					if ( Key[ 0 ] == threadIdx.x )
					{
						Penalty = BMDist;
						if ( BestPlace < N )
						{
							temp = Cust[ PCust[ BestPlace + Sh ] ];

							TW[ CEP + Sh ] = fmaxf( TW[ temp + Sh ] + Ser + CustDist[ temp * ( N + 1 ) + CEP ], Red[ CEP ] );
							temp = Cust[ Car[ BestPlace + Sh ] + N ];
							Arrival = fmaxf( CustDist[ N * ( N + 1 ) + temp ], Red[ temp ] );
							while ( temp != N )
							{
								Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + Cust[ temp ] ], Red[ Cust[ temp ] ] );
								temp = Cust[ temp ];
							}
						}
						else
						{
							TW[ CEP + Sh ] = fmaxf( CustDist[ ( N ) * ( N + 1 ) + CEP ], Red[ CEP ] );
						}
						Cust[ CEP ] = Cust[ BestPlace ];
						PCust[ CEP + Sh ] = BestPlace;
						if ( Cust[ BestPlace ] < N )
						{
							PCust[ Cust[ BestPlace ] + Sh ] = CEP;
						}
						if ( BestPlace < N )
						{
							Car[ CEP + Sh ] = Car[ BestPlace + Sh ];
						}
						else
						{
							Car[ CEP + Sh ] = Car[ Cust[ BestPlace ] + Sh ];
						}
						CarLoad[ Car[ CEP + Sh ] + Sh ] += Dem[ CEP ];
						Cust[ BestPlace ] = CEP;
						temp = Cust[ CEP ];
						Arrival = fmaxf( TW[ CEP + Sh ] + Ser + CustDist[ CEP * ( N + 1 ) + temp ], Red[ temp ] );
						temp2 = temp;
						while ( temp2 < N && Arrival != TW[ temp2 + Sh ] )
						{
							TW[ temp2 + Sh ] = Arrival;
							temp = temp2;
							temp2 = Cust[ temp2 ];
							Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
						}
					}
				}
				__syncthreads();
				/**************************************/
				/**Sort the most promising Customers**/
				/************************************/
				found = true;
				int BTotalWeight;
				int BeginCar = Car[ CEP + Sh ] + N;
				int Counter;
				while ( Penalty > 0 && found )
				{
					temp = Cust[ BeginCar ];
					Counter = 0;
					possible = true;
					if ( idx < 96 )
					{
						Prom[ idx ] = 0;
					}
					while ( temp != N && idx > Counter && idx < 96 )
					{
						temp = Cust[ temp ];
						Counter++;
					}
					if ( idx > Counter )
					{
						k = Counter;
					}
					else
					{
						temp3 = temp;
					}
					if ( temp < N && idx < 96 && temp != CEP )
					{
						if ( CarLoad[ Car[ CEP + Sh ] + Sh ] > MaxLoad )
						{
							Prom[ idx ] += fminf( Dem[ temp ], CarLoad[ BeginCar - N + Sh ] - MaxLoad );
							KeyProm[ idx ] = temp3;
						}
						temp2 = Cust[ temp ];
						if ( idx == 0 )
						{
							Arrival = fmaxf( CustDist[ N * ( N + 1 ) + temp2 ], Red[ temp2 ] );
						}
						else
						{
							if ( temp2 < N )
							{
								Arrival = fmaxf( TW[ PCust[ temp + Sh ] + Sh ] + Ser + CustDist[ PCust[ temp + Sh ] * ( N + 1 ) + temp2 ], Red[ temp2 ] );
								if ( TW[ temp + Sh ] > Due[ temp ] )
								{
									Prom[ idx ] += TW[ temp + Sh ] - Due[ temp ];
									KeyProm[ idx ] = temp3;
								}
							}
							else
							{
								temp = PCust[ temp + Sh ];
								Arrival = TW[ temp + Sh ] + Ser + CustDist[ temp * ( N + 1 ) + N ];
								temp = Cust[ temp ];
								if ( TW[ temp + Sh ] > Due[ temp ] )
								{
									Prom[ idx ] += TW[ temp + Sh ] - Due[ temp ];
									KeyProm[ idx ] = temp3;
								}
							}
						}
						while ( temp2 < N && Arrival != TW[ temp2 + Sh ] )
						{
							if ( TW[ temp2 + Sh ] > Due[ temp2 ] )
							{
								Prom[ idx ] += fminf( TW[ temp2 + Sh ] - Arrival, TW[ temp2 + Sh ] - Due[ temp2 ] );
								KeyProm[ idx ] = temp3;
							}
							temp = temp2;
							temp2 = Cust[ temp ];
							Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
						}
					}
					__syncthreads();
					if ( k < 64 && idx < 32 )
					{
						float ftemp;
						for ( int size = 2; size <= 64; size <<= 1 )
						{
							for ( int stride = size >> 1; stride > 0; stride >>= 1 )
							{
								temp2 = ( idx + ( idx & ~( stride - 1 ) ) ) + ( ( idx & size >> 1 ) > 0 ) * stride;
								temp3 = ( idx + ( idx & ~( stride - 1 ) ) ) + ( 1 - ( ( idx & size >> 1 ) > 0 ) ) * stride;
								if ( Prom[ temp2 ] < Prom[ temp3 ] )
								{
									ftemp = Prom[ temp2 ];

									Prom[ temp2 ] = Prom[ temp3 ];
									Prom[ temp3 ] = ftemp;
									temp = KeyProm[ temp2 ];
									KeyProm[ temp2 ] = KeyProm[ temp3 ];
									KeyProm[ temp3 ] = temp;
								}
							}
						}
					}
					/*******************************************************/
					/**Decrease Penalty with 1 move from infeasible route**/
					/*****************************************************/

					int WLocal;
					float BLpenalty;
					found = false;
					int z = 0;
					Best[ idx ] = 10000.0f;
					BestPlace = -1;
					__syncthreads();
					while ( ( !found || Prom[ z ] == Prom[ z - 1 ] ) && Prom[ z ] > 0 )
					{
						temp3 = KeyProm[ z ];
						/*if (temp3 - 226 >= 0)
						{
							idx = temp3 - 226 + threadIdx.x;
						}
						else
						{
							idx = threadIdx.x;
						}*/
						idx = BestCust[ temp3 * blockDim.x + threadIdx.x ];
						__syncthreads();
						if ( idx < N )
						{
							possible = false;
							if ( Cust[ idx ] >= 0 && CarLoad[ Car[ idx + Sh ] + Sh ] + Dem[ temp3 ] <= MaxLoad && Car[ idx + Sh ] != Car[ temp3 + Sh ] )
							{
								temp = idx;
								Arrival = fmaxf( TW[ temp + Sh ] + Ser + CustDist[ temp3 * ( N + 1 ) + temp ], Red[ temp3 ] );
								if ( Arrival <= Due[ temp3 ] )
								{
									temp2 = Cust[ idx ];
									possible = true;
									Arrival = fmaxf( Arrival + Ser + CustDist[ temp3 * ( N + 1 ) + temp2 ], Red[ temp2 ] );
								}
								while ( possible && temp2 < N && Arrival > TW[ temp2 + Sh ] )
								{
									if ( Arrival > Due[ temp2 ] )
									{
										possible = false;
									}
									else
									{
										temp = temp2;
										temp2 = Cust[ temp2 ];
										Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
									}
								}
								if ( possible )
								{
									if ( PCust[ temp3 + Sh ] < N )
									{
										MDist = CustDist[ PCust[ temp3 + Sh ] * ( N + 1 ) + Cust[ temp3 ] ] - ( CustDist[ PCust[ temp3 + Sh ] * ( N + 1 ) + temp3 ] + CustDist[ temp3 * ( N + 1 ) + Cust[ temp3 ] ] );
									}
									else
									{
										MDist = CustDist[ N * ( N + 1 ) + Cust[ temp3 ] ] - ( CustDist[ N * ( N + 1 ) + temp3 ] + CustDist[ temp3 * ( N + 1 ) + Cust[ temp3 ] ] );
									}
									MDist += CustDist[ idx * ( N + 1 ) + temp3 ] + CustDist[ temp3 * ( N + 1 ) + Cust[ idx ] ] - CustDist[ idx * ( N + 1 ) + Cust[ idx ] ];
									if ( MDist < Best[ threadIdx.x ] )
									{
										Best[ threadIdx.x ] = MDist;
										BestPlace = idx;
										temp4 = temp3;
										Key[ threadIdx.x ] = threadIdx.x;
										found = true;
										WLocal = 0;
									}
								}
							}
						}
						if ( temp3 < 250 )
						{
							idx = N + threadIdx.x;
							while ( idx < N + NCars )
							{
								if ( CarLoad[ idx - N + Sh ] + Dem[ temp3 ] <= MaxLoad && idx - N != Car[ temp3 + Sh ] )
								{
									temp = Cust[ idx ];
									Arrival = fmaxf( CustDist[ ( N ) * ( N + 1 ) + temp3 ], Red[ temp3 ] );
									Arrival = fmaxf( Arrival + Ser + CustDist[ temp3 * ( N + 1 ) + temp ], Red[ temp ] );
									temp2 = temp;
									possible = true;
									while ( possible && temp2 < N && Arrival > TW[ temp2 + Sh ] )
									{
										if ( Arrival > Due[ temp2 ] )
										{
											possible = false;
										}
										else
										{
											temp = temp2;
											temp2 = Cust[ temp2 ];
											if ( temp2 == temp3 )
											{
												temp2 = Cust[ temp2 ];
											}
											Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
										}
									}
									if ( possible )
									{
										if ( PCust[ temp3 + Sh ] < N )
										{
											MDist = CustDist[ PCust[ temp3 + Sh ] * ( N + 1 ) + Cust[ temp3 ] ] - ( CustDist[ PCust[ temp3 + Sh ] * ( N + 1 ) + temp3 ] + CustDist[ temp3 * ( N + 1 ) + Cust[ temp3 ] ] );
										}
										else
										{
											MDist = CustDist[ N * ( N + 1 ) + Cust[ temp3 ] ] - ( CustDist[ N * ( N + 1 ) + temp3 ] + CustDist[ temp3 * ( N + 1 ) + Cust[ temp3 ] ] );
										}
										MDist += MDist = CustDist[ N * ( N + 1 ) + temp3 ] + CustDist[ temp3 * ( N + 1 ) + Cust[ idx ] ] - CustDist[ N * ( N + 1 ) + Cust[ idx ] ];
										if ( MDist < Best[ threadIdx.x ] )
										{
											Best[ threadIdx.x ] = MDist;
											BestPlace = idx;
											Key[ threadIdx.x ] = threadIdx.x;
											temp4 = temp3;
											found = true;
											WLocal = 0;
										}
									}
								}
								idx += blockDim.x;
							}
						}
						/****************/
						/*SWAP MOVEMENT*/
						/**************/
						/*if (temp3 - 226 >= 0)
						{
							idx = temp3 - 226 + threadIdx.x;
						}
						else
						{
							idx = threadIdx.x;
						}*/
						idx = BestCust[ temp3 * blockDim.x + threadIdx.x ];
						if ( idx < N )
						{
							if ( Cust[ idx ] < N && Cust[ idx ] >= 0 && ( ( CarLoad[ Car[ idx + Sh ] + Sh ] + Dem[ temp3 ] - Dem[ Cust[ idx ] ] <= MaxLoad && CarLoad[ Car[ temp3 + Sh ] + Sh ] + Dem[ Cust[ idx ] ] - Dem[ temp3 ] <= MaxLoad ) && Car[ idx + Sh ] != Car[ temp3 + Sh ] ) )
							{
								possible = false;
								LPenalty = 0;
								temp = idx;
								Arrival = fmaxf( TW[ temp + Sh ] + Ser + CustDist[ temp3 * ( N + 1 ) + temp ], Red[ temp3 ] );
								if ( Arrival <= Due[ temp3 ] )
								{
									possible = true;
									temp = temp3;
									temp2 = Cust[ Cust[ idx ] ];
									Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
								}
								while ( possible && temp2 < N && Arrival > TW[ temp2 + Sh ] )
								{
									if ( Arrival > Due[ temp2 ] )
									{
										possible = false;
									}
									else
									{
										temp = temp2;
										temp2 = Cust[ temp2 ];
										Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
									}
								}
								if ( possible )
								{
									temp = PCust[ temp3 + Sh ];
									temp2 = Cust[ idx ];
									if ( CarLoad[ Car[ temp3 + Sh ] + Sh ] > MaxLoad )
									{
										LPenalty += fminf( Dem[ temp3 ] - Dem[ temp2 ], CarLoad[ Car[ temp3 + Sh ] + Sh ] - MaxLoad );
									}
									if ( temp > N )
									{
										Arrival = fmaxf( CustDist[ N * ( N + 1 ) + temp2 ], Red[ temp2 ] );
									}
									else
									{
										Arrival = fmaxf( TW[ temp + Sh ] + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
									}

									LPenalty += ( fmaxf( TW[ temp3 + Sh ] - Due[ temp3 ], 0 ) - fmaxf( Arrival - Due[ temp2 ], 0 ) );

									temp = temp2;
									temp2 = Cust[ temp3 ];
									Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
									while ( ( LPenalty >= 0 || Arrival < TW[ temp2 + Sh ] ) && temp2 != N && Arrival != TW[ temp2 + Sh ] )
									{
										LPenalty += fmaxf( TW[ temp2 + Sh ] - Due[ temp2 ], 0 ) - fmaxf( Arrival - Due[ temp2 ], 0 );
										temp = temp2;
										temp2 = Cust[ temp2 ];
										Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
									}
									if ( LPenalty > 0 )
									{
										temp2 = Cust[ idx ];
										if ( PCust[ temp3 + Sh ] < N )
										{
											MDist = CustDist[ PCust[ temp3 + Sh ] * ( N + 1 ) + temp2 ] + CustDist[ temp2 * ( N + 1 ) + Cust[ temp3 ] ] - ( CustDist[ PCust[ temp3 + Sh ] * ( N + 1 ) + temp3 ] + CustDist[ temp3 * ( N + 1 ) + Cust[ temp3 ] ] );
										}
										else
										{
											MDist = CustDist[ N * ( N + 1 ) + temp2 ] + CustDist[ temp2 * ( N + 1 ) + Cust[ temp3 ] ] - ( CustDist[ N * ( N + 1 ) + temp3 ] + CustDist[ temp3 * ( N + 1 ) + Cust[ temp3 ] ] );

										}

										MDist += CustDist[ idx * ( N + 1 ) + temp3 ] + CustDist[ temp3 * ( N + 1 ) + Cust[ temp2 ] ] - ( CustDist[ idx * ( N + 1 ) + temp2 ] + CustDist[ temp2 * ( N + 1 ) + Cust[ temp2 ] ] );
										MDist += 1000 * ( 1 - LPenalty / Prom[ z ] );
										if ( MDist < Best[ threadIdx.x ] )
										{
											Best[ threadIdx.x ] = MDist;
											BestPlace = idx;
											temp4 = temp3;
											Key[ threadIdx.x ] = threadIdx.x;
											BLpenalty = LPenalty;
											found = true;
											WLocal = 1;
										}
									}
								}
							}
						}
						if ( temp3 < 250 )
						{
							idx = N + threadIdx.x;
							while ( idx < N + NCars )
							{
								LPenalty = 0;
								if ( ( CarLoad[ idx - N + Sh ] + Dem[ temp3 ] - Dem[ Cust[ idx ] ] <= MaxLoad && CarLoad[ Car[ temp3 + Sh ] + Sh ] + Dem[ Cust[ idx ] ] - Dem[ temp3 ] <= MaxLoad ) && idx - N != Car[ temp3 + Sh ] )
								{
									temp = Cust[ idx ];
									Arrival = fmaxf( CustDist[ ( N ) * ( N + 1 ) + temp3 ], Red[ temp3 ] );
									temp2 = Cust[ temp ];
									Arrival = fmaxf( Arrival + Ser + CustDist[ temp3 * ( N + 1 ) + temp2 ], Red[ temp2 ] );
									possible = true;
									while ( possible && temp2 < N && Arrival > TW[ temp2 + Sh ] )
									{
										if ( Arrival > Due[ temp2 ] )
										{
											possible = false;
										}
										else
										{
											temp = temp2;
											temp2 = Cust[ temp2 ];
											Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
										}
									}
									if ( possible )
									{
										temp = PCust[ temp3 + Sh ];
										temp2 = Cust[ idx ];
										if ( CarLoad[ Car[ temp3 + Sh ] + Sh ] > MaxLoad )
										{
											LPenalty += fminf( Dem[ temp3 ] - Dem[ temp2 ], CarLoad[ Car[ temp3 + Sh ] + Sh ] - MaxLoad );
										}
										if ( temp > N )
										{
											Arrival = fmaxf( CustDist[ N * ( N + 1 ) + temp2 ], Red[ temp2 ] );
										}
										else
										{
											Arrival = fmaxf( TW[ temp + Sh ] + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
										}
										LPenalty += ( fmaxf( TW[ temp3 + Sh ] - Due[ temp3 ], 0 ) - fmaxf( Arrival - Due[ temp2 ], 0 ) );
										temp = temp2;
										temp2 = Cust[ temp3 ];
										Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
										while ( ( LPenalty >= 0 || Arrival < TW[ temp2 + Sh ] ) && temp2 != N && Arrival != TW[ temp2 + Sh ] )
										{
											LPenalty += fmaxf( TW[ temp2 + Sh ] - Due[ temp2 ], 0 ) - fmaxf( Arrival - Due[ temp2 ], 0 );
											temp = temp2;
											temp2 = Cust[ temp2 ];
											Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
										}
										if ( LPenalty > 0 )
										{
											temp2 = Cust[ idx ];
											if ( PCust[ temp3 + Sh ] < N )
											{
												MDist = CustDist[ PCust[ temp3 + Sh ] * ( N + 1 ) + temp2 ] + CustDist[ temp2 * ( N + 1 ) + Cust[ temp3 ] ] - ( CustDist[ PCust[ temp3 + Sh ] * ( N + 1 ) + temp3 ] + CustDist[ temp3 * ( N + 1 ) + Cust[ temp3 ] ] );
											}
											else
											{
												MDist = CustDist[ N * ( N + 1 ) + temp2 ] + CustDist[ temp2 * ( N + 1 ) + Cust[ temp3 ] ] - ( CustDist[ N * ( N + 1 ) + temp3 ] + CustDist[ temp3 * ( N + 1 ) + Cust[ temp3 ] ] );
											}
											MDist += CustDist[ N * ( N + 1 ) + temp3 ] + CustDist[ temp3 * ( N + 1 ) + Cust[ temp2 ] ] - ( CustDist[ N * ( N + 1 ) + temp2 ] + CustDist[ temp2 * ( N + 1 ) + Cust[ temp2 ] ] );
											MDist = 1000 * ( 1 - LPenalty / Prom[ z ] );
											if ( MDist < Best[ threadIdx.x ] )
											{
												Best[ threadIdx.x ] = MDist;
												BestPlace = idx;
												temp4 = temp3;
												Key[ threadIdx.x ] = threadIdx.x;
												BLpenalty = LPenalty;
												found = true;
												WLocal = 1;
											}
										}
									}
								}
								idx += blockDim.x;
							}
						}
						/*****************/
						/*2-opt Movement*/
						/***************/
						int TotalWeight = 0;
						/*if (temp3 - 226 >= 0)
						{
							idx = temp3 - 226 + threadIdx.x;
						}
						else
						{
							idx = threadIdx.x;
						}*/
						idx = BestCust[ temp3 * blockDim.x + threadIdx.x ];
						if ( idx < N )
						{
							if ( Cust[ idx ] >= 0 && Car[ idx + Sh ] != Car[ temp3 + Sh ] )
							{
								TotalWeight = 0;
								LPenalty = 0;
								possible = false;
								Arrival = fmaxf( TW[ idx + Sh ] + Ser + CustDist[ temp3 * ( N + 1 ) + idx ], Red[ temp3 ] );
								if ( Arrival <= Due[ temp3 ] )
								{
									TotalWeight += Dem[ idx ] + Dem[ temp3 ];
									possible = true;
									temp = temp3;
									temp2 = Cust[ temp3 ];
									Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
								}
								while ( possible && temp2 < N && Arrival > TW[ temp2 + Sh ] )
								{
									if ( Arrival > Due[ temp2 ] )
									{
										possible = false;
									}
									else
									{
										temp = temp2;
										TotalWeight += Dem[ temp ];
										temp2 = Cust[ temp2 ];
										Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
									}
								}
								temp = PCust[ idx + Sh ];
								while ( possible && ( temp2 < N || temp < N ) )
								{
									if ( temp2 < N )
									{
										TotalWeight += Dem[ temp2 ];
										temp2 = Cust[ temp2 ];
									}
									if ( temp < N )
									{
										TotalWeight += Dem[ temp ];
										temp = PCust[ temp + Sh ];
									}

								}
								if ( possible && TotalWeight <= MaxLoad )
								{
									temp = PCust[ temp3 + Sh ];
									temp2 = Cust[ idx ];
									if ( CarLoad[ Car[ temp3 + Sh ] + Sh ] + CarLoad[ Car[ idx + Sh ] + Sh ] - TotalWeight > MaxLoad )
									{
										LPenalty -= CarLoad[ Car[ temp3 + Sh ] + Sh ] + CarLoad[ Car[ idx + Sh ] + Sh ] - TotalWeight - MaxLoad - fmaxf( CarLoad[ Car[ temp3 + Sh ] + Sh ] - MaxLoad, 0 );
									}
									else
									{
										LPenalty += fmaxf( CarLoad[ Car[ temp3 + Sh ] + Sh ] - MaxLoad, 0 );
									}
									if ( temp >= N )
									{
										Arrival = fmaxf( CustDist[ N * ( N + 1 ) + temp2 ], Red[ temp2 ] );
									}
									else
									{
										Arrival = fmaxf( TW[ temp + Sh ] + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
									}
									LPenalty += ( fmaxf( TW[ temp3 + Sh ] - Due[ temp3 ], 0 ) - fmaxf( Arrival - Due[ temp2 ], 0 ) );
									if ( temp2 < N )
									{
										temp = temp2;
										temp2 = Cust[ temp2 ];
										Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
									}
									while ( ( LPenalty >= 0 || Arrival < TW[ temp2 + Sh ] ) && temp2 < N && Arrival != TW[ temp2 + Sh ] )
									{
										LPenalty += fmaxf( TW[ temp2 + Sh ] - Due[ temp2 ], 0 ) - fmaxf( Arrival - Due[ temp2 ], 0 );
										temp = temp2;
										temp2 = Cust[ temp2 ];
										Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
									}
									if ( LPenalty > 0 )
									{
										if ( PCust[ temp3 + Sh ] < N )
										{
											MDist = CustDist[ PCust[ temp3 + Sh ] * ( N + 1 ) + Cust[ idx ] ] + CustDist[ temp3 * ( N + 1 ) + idx ] - ( CustDist[ PCust[ temp3 + Sh ] * ( N + 1 ) + temp3 ] + CustDist[ idx * ( N + 1 ) + Cust[ idx ] ] );
										}
										else
										{
											MDist = CustDist[ N * ( N + 1 ) + Cust[ idx ] ] + CustDist[ temp3 * ( N + 1 ) + idx ] - ( CustDist[ N * ( N + 1 ) + temp3 ] + CustDist[ idx * ( N + 1 ) + Cust[ idx ] ] );
										}
										MDist += 1000 * ( 1 - LPenalty / Prom[ z ] );
										if ( MDist < Best[ threadIdx.x ] )
										{
											Best[ threadIdx.x ] = MDist;
											BestPlace = idx;
											temp4 = temp3;
											Key[ threadIdx.x ] = threadIdx.x;
											BLpenalty = LPenalty;
											found = true;
											WLocal = 2;
											BTotalWeight = TotalWeight;
										}
									}
								}
							}
						}
						if ( temp3 < 250 )
						{
							idx = N + threadIdx.x;
							while ( idx < N + NCars )
							{
								if ( idx - N != Car[ temp3 + Sh ] )
								{
									TotalWeight = 0;
									LPenalty = 0;
									possible = false;
									Arrival = fmaxf( CustDist[ N * ( N + 1 ) + temp3 ], Red[ temp3 ] );
									if ( Arrival <= Due[ temp3 ] )
									{
										TotalWeight += Dem[ temp3 ];
										possible = true;
										temp = temp3;
										temp2 = Cust[ temp3 ];
										Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
									}
									while ( possible && temp2 != N && Arrival > TW[ temp2 + Sh ] )
									{
										if ( Arrival > Due[ temp2 ] )
										{
											possible = false;
										}
										else
										{
											temp = temp2;
											TotalWeight += Dem[ temp ];
											temp2 = Cust[ temp2 ];
											Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
										}
									}
									while ( possible && temp2 < N )
									{
										if ( temp2 < N )
										{
											TotalWeight += Dem[ temp2 ];
											temp2 = Cust[ temp2 ];
										}
									}
									if ( possible && TotalWeight <= MaxLoad )
									{
										temp = PCust[ temp3 + Sh ];
										temp2 = Cust[ idx ];
										if ( CarLoad[ Car[ temp3 + Sh ] + Sh ] + CarLoad[ idx - N + Sh ] - TotalWeight > MaxLoad )
										{
											LPenalty -= CarLoad[ Car[ temp3 + Sh ] + Sh ] + CarLoad[ idx - N + Sh ] - TotalWeight - MaxLoad - fmaxf( CarLoad[ Car[ temp3 + Sh ] + Sh ] - MaxLoad, 0 );
										}
										else
										{
											LPenalty += fmaxf( CarLoad[ Car[ temp3 + Sh ] + Sh ] - MaxLoad, 0 );
										}
										if ( temp < N )
										{
											Arrival = fmaxf( TW[ temp + Sh ] + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
										}
										else
										{
											Arrival = fmaxf( CustDist[ N * ( N + 1 ) + temp2 ], Red[ temp2 ] );
										}
										LPenalty += ( fmaxf( TW[ temp3 + Sh ] - Due[ temp3 ], 0 ) - fmaxf( Arrival - Due[ temp2 ], 0 ) );
										if ( temp2 < N )
										{
											temp = temp2;
											temp2 = Cust[ temp2 ];
											Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
										}
										while ( ( LPenalty >= 0 || Arrival < TW[ temp2 + Sh ] ) && temp2 < N && Arrival != TW[ temp2 + Sh ] )
										{
											LPenalty += fmaxf( TW[ temp2 + Sh ] - Due[ temp2 ], 0 ) - fmaxf( Arrival - Due[ temp2 ], 0 );
											temp = temp2;
											temp2 = Cust[ temp2 ];
											Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
										}
										if ( LPenalty > 0 )
										{
											if ( PCust[ temp3 + Sh ] < N )
											{
												MDist = CustDist[ PCust[ temp3 + Sh ] * ( N + 1 ) + Cust[ idx ] ] + CustDist[ temp3 * ( N + 1 ) + N ] - ( CustDist[ PCust[ temp3 + Sh ] * ( N + 1 ) + temp3 ] + CustDist[ N * ( N + 1 ) + Cust[ idx ] ] );
											}
											else
											{
												MDist = CustDist[ N * ( N + 1 ) + Cust[ idx ] ] + CustDist[ temp3 * ( N + 1 ) + N ] - ( CustDist[ N * ( N + 1 ) + temp3 ] + CustDist[ N * ( N + 1 ) + Cust[ idx ] ] );
											}

											MDist += 1000 * ( 1 - LPenalty / Prom[ z ] );
											if ( MDist < Best[ threadIdx.x ] )
											{
												Best[ threadIdx.x ] = MDist;
												BestPlace = idx;
												temp4 = temp3;
												Key[ threadIdx.x ] = threadIdx.x;
												BLpenalty = LPenalty;
												found = true;
												WLocal = 2;
												BTotalWeight = TotalWeight;
											}
										}
									}
								}
								idx += blockDim.x;
							}
						}
						z += 1;
						__syncthreads();
					}
					__syncthreads();
					idx = threadIdx.x;
					//			if (idx < 512){ if (Best[idx] > Best[idx + 512]){ Best[idx] = Best[idx + 512]; Key[idx] = Key[idx + 512]; } } __syncthreads();
					//if (idx < 256){ if (Best[idx] > Best[idx + 256]){ Best[idx] = Best[idx + 256]; Key[idx] = Key[idx + 256]; } } __syncthreads();
					if ( idx < 128 )
					{
						if ( Best[ idx ] > Best[ idx + 128 ] )
						{
							Best[ idx ] = Best[ idx + 128 ]; Key[ idx ] = Key[ idx + 128 ];
						}
					} __syncthreads();
					if ( idx < 64 )
					{
						if ( Best[ idx ] > Best[ idx + 64 ] )
						{
							Best[ idx ] = Best[ idx + 64 ]; Key[ idx ] = Key[ idx + 64 ];
						}
					} __syncthreads();
					if ( idx < 32 )
					{
						if ( Best[ idx ] > Best[ idx + 32 ] )
						{
							Best[ idx ] = Best[ idx + 32 ]; Key[ idx ] = Key[ idx + 32 ];
						}
						if ( Best[ idx ] > Best[ idx + 16 ] )
						{
							Best[ idx ] = Best[ idx + 16 ]; Key[ idx ] = Key[ idx + 16 ];
						}
						if ( Best[ idx ] > Best[ idx + 8 ] )
						{
							Best[ idx ] = Best[ idx + 8 ]; Key[ idx ] = Key[ idx + 8 ];
						}
						if ( Best[ idx ] > Best[ idx + 4 ] )
						{
							Best[ idx ] = Best[ idx + 4 ]; Key[ idx ] = Key[ idx + 4 ];
						}
						if ( Best[ idx ] > Best[ idx + 2 ] )
						{
							Best[ idx ] = Best[ idx + 2 ]; Key[ idx ] = Key[ idx + 2 ];
						}
						if ( Best[ idx ] > Best[ idx + 1 ] )
						{
							Best[ idx ] = Best[ idx + 1 ]; Key[ idx ] = Key[ idx + 1 ];
						}
					}
					__syncthreads();
					if ( found )
					{
						if ( Key[ 0 ] == threadIdx.x )
						{
							temp3 = temp4;
							if ( WLocal == 0 )
							{
								Penalty -= Prom[ z - 1 ];
								if ( Penalty <= 0 )
								{
									j++;
								}
								/********************/
								/**Delete Customer**/
								/******************/
								Cust[ PCust[ temp3 + Sh ] ] = Cust[ temp3 ];
								if ( Cust[ temp3 ] < N )
								{
									PCust[ Cust[ temp3 ] + Sh ] = PCust[ temp3 + Sh ];
								}
								CarLoad[ Car[ temp3 + Sh ] + Sh ] -= Dem[ temp3 ];
								temp = PCust[ temp3 + Sh ];
								temp2 = Cust[ temp ];
								if ( temp < N )
								{
									Arrival = fmaxf( TW[ temp + Sh ] + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
								}
								else
								{
									Arrival = fmaxf( CustDist[ N * ( N + 1 ) + temp2 ], Red[ temp2 ] );
								}

								while ( temp2 < N && Arrival != TW[ temp2 + Sh ] )
								{
									TW[ temp2 + Sh ] = Arrival;
									temp = temp2;
									temp2 = Cust[ temp2 ];
									Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
								}
								/*****************/
								/**Add customer**/
								/***************/
								if ( BestPlace < N )
								{
									temp = Cust[ PCust[ BestPlace + Sh ] ];
									TW[ temp3 + Sh ] = fmaxf( TW[ temp + Sh ] + Ser + CustDist[ temp * ( N + 1 ) + temp3 ], Red[ temp3 ] );
								}
								else
								{
									TW[ temp3 + Sh ] = fmaxf( CustDist[ ( N ) * ( N + 1 ) + temp3 ], Red[ temp3 ] );
								}
								Cust[ temp3 ] = Cust[ BestPlace ];
								PCust[ temp3 + Sh ] = BestPlace;
								if ( Cust[ BestPlace ] < N )
								{
									PCust[ Cust[ BestPlace ] + Sh ] = temp3;
								}
								if ( BestPlace < N )
								{
									Car[ temp3 + Sh ] = Car[ BestPlace + Sh ];
								}
								else
								{
									Car[ temp3 + Sh ] = Car[ Cust[ BestPlace ] + Sh ];
								}
								CarLoad[ Car[ temp3 + Sh ] + Sh ] += Dem[ temp3 ];
								Cust[ BestPlace ] = temp3;
								temp = Cust[ temp3 ];
								Arrival = fmaxf( TW[ temp3 + Sh ] + Ser + CustDist[ temp3 * ( N + 1 ) + temp ], Red[ temp ] );
								temp2 = temp;
								while ( temp2 < N && Arrival != TW[ temp2 + Sh ] )
								{
									TW[ temp2 + Sh ] = Arrival;
									temp = temp2;
									temp2 = Cust[ temp2 ];
									Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
								}
							}
							else if ( WLocal == 1 )
							{
								Penalty -= BLpenalty;
								if ( Penalty <= 0 )
								{
									j++;
								}
								CarLoad[ Car[ temp3 + Sh ] + Sh ] = CarLoad[ Car[ temp3 + Sh ] + Sh ] - Dem[ temp3 ] + Dem[ Cust[ BestPlace ] ];
								if ( BestPlace < N )
								{
									CarLoad[ Car[ BestPlace + Sh ] + Sh ] = CarLoad[ Car[ BestPlace + Sh ] + Sh ] + Dem[ temp3 ] - Dem[ Cust[ BestPlace ] ];
								}
								else
								{
									CarLoad[ BestPlace - N + Sh ] = CarLoad[ BestPlace - N + Sh ] + Dem[ temp3 ] - Dem[ Cust[ BestPlace ] ];
								}
								temp4 = Car[ temp3 + Sh ];
								if ( BestPlace < N )
								{
									Car[ temp3 + Sh ] = Car[ BestPlace + Sh ];
								}
								else
								{
									Car[ temp3 + Sh ] = Car[ Cust[ BestPlace ] + Sh ];
								}
								Car[ Cust[ BestPlace ] + Sh ] = temp4;

								/********************/
								/**Swap Customer 1**/
								/******************/
								temp4 = Cust[ Cust[ BestPlace ] ];
								temp2 = Cust[ BestPlace ];
								Cust[ PCust[ temp3 + Sh ] ] = temp2;
								if ( Cust[ temp3 ] < N )
								{
									PCust[ Cust[ temp3 ] + Sh ] = temp2;
								}
								Cust[ temp2 ] = Cust[ temp3 ];
								PCust[ temp2 + Sh ] = PCust[ temp3 + Sh ];
								temp = PCust[ temp2 + Sh ];
								if ( temp < N )
								{
									TW[ temp2 + Sh ] = fmaxf( TW[ temp + Sh ] + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
								}
								else
								{
									TW[ temp2 + Sh ] = fmaxf( CustDist[ N * ( N + 1 ) + temp2 ], Red[ temp2 ] );
								}
								temp = temp2;
								temp2 = Cust[ temp2 ];
								Arrival = fmaxf( TW[ temp + Sh ] + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
								while ( temp2 < N && Arrival != TW[ temp2 + Sh ] )
								{
									TW[ temp2 + Sh ] = Arrival;
									temp = temp2;
									temp2 = Cust[ temp2 ];
									Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
								}
								/********************/
								/**Swap Customer 2**/
								/******************/
								if ( BestPlace < N )
								{
									temp = BestPlace;
									TW[ temp3 + Sh ] = fmaxf( TW[ temp + Sh ] + Ser + CustDist[ temp * ( N + 1 ) + temp3 ], Red[ temp3 ] );
								}
								else
								{
									TW[ temp3 + Sh ] = fmaxf( CustDist[ ( N ) * ( N + 1 ) + temp3 ], Red[ temp3 ] );
								}
								temp = temp4;
								Cust[ temp3 ] = temp;
								PCust[ temp3 + Sh ] = BestPlace;
								if ( temp < N )
								{
									PCust[ temp4 + Sh ] = temp3;
								}
								Cust[ BestPlace ] = temp3;
								temp = Cust[ temp3 ];
								Arrival = fmaxf( TW[ temp3 + Sh ] + Ser + CustDist[ temp3 * ( N + 1 ) + temp ], Red[ temp ] );
								temp2 = temp;
								while ( temp2 < N && Arrival != TW[ temp2 + Sh ] )
								{
									TW[ temp2 + Sh ] = Arrival;
									temp = temp2;
									temp2 = Cust[ temp2 ];
									Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
								}
							}
							else if ( WLocal == 2 )
							{
								Penalty -= BLpenalty;
								if ( Penalty <= 0 )
								{
									j++;
								}
								if ( BestPlace < N )
								{
									CarLoad[ Car[ temp3 + Sh ] + Sh ] += CarLoad[ Car[ BestPlace + Sh ] + Sh ];
									CarLoad[ Car[ BestPlace + Sh ] + Sh ] = BTotalWeight;
									CarLoad[ Car[ temp3 + Sh ] + Sh ] -= BTotalWeight;
								}
								else
								{
									CarLoad[ Car[ temp3 + Sh ] + Sh ] += CarLoad[ BestPlace - N + Sh ];
									CarLoad[ BestPlace - N + Sh ] = BTotalWeight;
									CarLoad[ Car[ temp3 + Sh ] + Sh ] -= BTotalWeight;
								}
								/********************/
								/**Swap Customer 1**/
								/******************/
								temp = PCust[ temp3 + Sh ];
								temp2 = Cust[ BestPlace ];
								Cust[ temp ] = temp2;
								temp4 = Car[ temp3 + Sh ];
								if ( temp2 < N )
								{
									PCust[ temp2 + Sh ] = temp;
									Car[ temp2 + Sh ] = temp4;
									if ( temp < N )
									{
										TW[ temp2 + Sh ] = fmaxf( TW[ temp + Sh ] + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
									}
									else
									{
										TW[ temp2 + Sh ] = fmaxf( CustDist[ N * ( N + 1 ) + temp2 ], Red[ temp2 ] );
									}
									temp = temp2;
									temp2 = Cust[ temp2 ];
									Arrival = fmaxf( TW[ temp + Sh ] + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
								}
								while ( temp2 < N && Arrival != TW[ temp2 + Sh ] )
								{
									Car[ temp2 + Sh ] = temp4;
									TW[ temp2 + Sh ] = Arrival;
									temp = temp2;
									temp2 = Cust[ temp2 ];
									Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
								}
								while ( temp2 < N )
								{
									Car[ temp2 + Sh ] = temp4;
									temp2 = Cust[ temp2 ];
								}
								/********************/
								/**Swap Customer 2**/
								/******************/
								Cust[ BestPlace ] = temp3;
								PCust[ temp3 + Sh ] = BestPlace;
								if ( BestPlace < N )
								{
									TW[ temp3 + Sh ] = fmaxf( TW[ BestPlace + Sh ] + Ser + CustDist[ BestPlace * ( N + 1 ) + temp3 ], Red[ temp3 ] );
									temp4 = Car[ BestPlace + Sh ];
								}
								else
								{
									TW[ temp3 + Sh ] = fmaxf( CustDist[ ( N ) * ( N + 1 ) + temp3 ], Red[ temp3 ] );
									temp4 = BestPlace - N;
								}
								temp = temp3;
								temp2 = Cust[ temp3 ];
								Car[ temp3 + Sh ] = temp4;
								Arrival = fmaxf( TW[ temp + Sh ] + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
								while ( temp2 < N && Arrival != TW[ temp2 + Sh ] )
								{
									Car[ temp2 + Sh ] = temp4;
									TW[ temp2 + Sh ] = Arrival;
									temp = temp2;
									temp2 = Cust[ temp2 ];
									Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
								}
								while ( temp2 < N )
								{
									Car[ temp2 + Sh ] = temp4;
									temp2 = Cust[ temp2 ];
								}
							}
						}
					}
					__syncthreads();
				}
				__syncthreads();


			}
{% endhighlight %}
</p>
</details>

As you can see this part has a lot of code, mostly redundant code I have to say. As far as I remember this part didn't really contribute to finding a better solution, so I will only explain the high-level details. The idea of this phase is to allow temporary an infeasibility in the solution. I insert the customer in the spot where it causes the least violation with respect to the capacity of the vehicle and the time windows. I try to solve this infeasibility by either moving a customer, swapping two customers or use a 2-opt algorithm. 

I wished I would have used functions. That would probably simplify this code by a lot.

<details><summary><div style="width:200px;height:25px;border:1px solid #999;">Click to show code!</div></summary>
<p>
{% highlight cpp linenos%}
/************************************/
			/*PART3: Penalty for hard customers*/
			/**********************************/
			if ( !found )
			{
				int Blex[ 3 ];
				int BPMax;
				float CEPGain;
				float CEPPen;
				bool Add;
				bool Shift;
				float MNGain[ RouteLength ];
				memset( MNGain, 0, sizeof( float ) * RouteLength );
				idx = threadIdx.x;
				while ( idx < N )
				{
					Cust[ idx ] = Custtemp[ idx + 2 * Sh ];
					Cust[ idx + N ] = Custtemp[ idx + 2 * Sh + N ];
					PCust[ idx + Sh ] = PCusttemp[ idx + Sh ];
					Car[ idx + Sh ] = Cartemp[ idx + Sh ];
					CarLoad[ idx + Sh ] = CarLoadtemp[ idx + Sh ];
					TW[ idx + Sh ] = TWtemp[ idx + blockIdx.x * N ];
					idx += blockDim.x;
				}
				idx = threadIdx.x;
				if ( idx == 0 )
				{
					CustPen[ CEP + Sh ]++;
				}
				PBest = CustPen[ CEP + Sh ];
				Blex[ 0 ] = CEP;
				__syncthreads();
				possible = false;
				if ( Cust[ idx ] >= 0 )
				{
					Best[ idx ] = CustPen[ CEP + Sh ];
					Key[ idx ] = threadIdx.x;
				}
				else
				{
					Best[ idx ] = CustPen[ CEP + Sh ] + 1;
				}
				BestPlace = idx;
				BPMax = 1;

				bool BehindVehicle = true;
				bool firstround = true;
				/*if (CEP - 226 >= 0)
				{
					idx = CEP - 226 + threadIdx.x;
					if (idx >= N)
					{
						BehindVehicle = false;
					}

				}
				else
				{
					idx = threadIdx.x;
					if (idx >= N)
					{
						BehindVehicle = false;
					}
				}*/
				idx = BestCust[ temp3 * blockDim.x + threadIdx.x ];
				while ( BehindVehicle )
				{
					bool Rdy = false;
					int lex[ 3 ];
					int PLex = 1;
					int PMax = 0;
					float PGain[ 3 ];
					float PPen[ 3 ];
					int h = 0;
					float CepTW;
					int tempPSum;

					if ( idx < N && Cust[ idx ] >= 0 )
					{
						temp = idx;
						temp2 = Cust[ idx ];
						Arrival = fmaxf( TW[ temp + Sh ] + Ser + CustDist[ CEP * ( N + 1 ) + temp ], Red[ CEP ] );
						CepTW = Arrival;
						MNGain[ h ] = fmaxf( Arrival - Due[ CEP ], 0 );
						CEPPen = fmaxf( Arrival - Due[ CEP ], 0.0f );
						CEPGain = fmaxf( Arrival - Red[ CEP ], 0.0f );
						temp = CEP;
						Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
						h++;
						while ( temp != N && Arrival != TW[ temp2 + Sh ] )
						{
							MNGain[ h ] = fmaxf( Arrival - Due[ temp2 ], 0 );
							int i = 1;
							while ( h - i >= 0 && MNGain[ h ] > MNGain[ h - i ] )
							{
								MNGain[ h - i ] = MNGain[ h ];
								i++;
							}
							temp = temp2;
							temp2 = Cust[ temp2 ];
							Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
							h++;
						}
					}

					else if ( idx >= N && idx < N + NCars && Cust[ idx ] >= 0 )
					{
						temp = Cust[ idx ];
						Arrival = fmaxf( CustDist[ ( N ) * ( N + 1 ) + CEP ], Red[ CEP ] );
						CepTW = Arrival;
						MNGain[ h ] = fmaxf( Arrival - Due[ CEP ], 0 );
						h++;
						CEPPen = fmaxf( Arrival - Due[ CEP ], 0.0f );
						CEPGain = fmaxf( Arrival - Red[ CEP ], 0.0f );
						Arrival = fmaxf( Arrival + Ser + CustDist[ CEP * ( N + 1 ) + temp ], Red[ temp ] );
						temp2 = temp;
						while ( temp != N && Arrival != TW[ temp2 + Sh ] )
						{
							MNGain[ h ] = fmaxf( Arrival - Due[ temp2 ], 0 );
							int i = 1;
							while ( h - i >= 0 && MNGain[ h ] > MNGain[ h - i ] )
							{
								MNGain[ h - i ] = MNGain[ h ];
								i++;
							}
							temp = temp2;
							temp2 = Cust[ temp2 ];
							Arrival = Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ];
							h++;
						}
					}
					int CarLoadCep;
					if ( Cust[ idx ] >= 0 )
					{

						if ( idx < N )
						{
							CarLoadCep = CarLoad[ Car[ idx + Sh ] + Sh ] + Dem[ CEP ];
						}
						else
						{
							CarLoadCep = CarLoad[ idx - N + Sh ] + Dem[ CEP ];
							if ( CEPPen > 0 )
							{
								Rdy = true;
							}
						}
						int load = CarLoadCep;
						Arrival = CepTW;
						int PSum = 0;
						float Gain = 0;
						int KMax = 2;
						while ( !Rdy )
						{
							if ( PSum < PBest )
							{
								if ( PMax > 0 )
								{
									load = CarLoadCep;
									for ( int i = 0; i < PMax; i++ )
									{
										load -= Dem[ lex[ i ] ];
									}
								}
								if ( Arrival <= Due[ CEP ] && Gain >= MNGain[ 0 ] )
								{
									if ( load <= MaxLoad )
									{
										Best[ threadIdx.x ] = PSum;
										PBest = PSum;
										Key[ threadIdx.x ] = threadIdx.x;
										BestPlace = idx;
										BPMax = PMax;
										for ( int i = 0; i < PMax; i++ )
										{
											Blex[ i ] = lex[ i ];
										}
									}
									else if ( PMax <= KMax )
									{
										if ( idx < N )
										{
											temp = Cust[ Car[ idx + Sh ] + N ];
										}
										else
										{
											temp = Cust[ idx ];
										}
										int k1 = 0;
										while ( temp == lex[ k1 ] && k1 < PMax )
										{
											temp = Cust[ temp ];
											k1++;
										}
										tempPSum = PSum;
										if ( PMax == 0 )
										{
											Rdy == true;
										}

										while ( temp < N )
										{
											int tempload = load - Dem[ temp ];
											tempPSum = PSum + CustPen[ temp + Sh ];
											if ( tempPSum < PBest && tempload <= MaxLoad )
											{
												PBest = tempPSum;
												Best[ threadIdx.x ] = tempPSum;
												Blex[ PMax ] = temp;
												Key[ threadIdx.x ] = threadIdx.x;
												BestPlace = idx;
												BPMax = PMax + 1;
												for ( int i = 0; i < PMax; i++ )
												{
													Blex[ i ] = lex[ i ];
												}
											}
											else if ( PMax < KMax && tempPSum < PBest - 1 )
											{
												int k2 = k1;
												temp2 = Cust[ temp ];
												while ( temp2 == lex[ k2 ] && k2 < PMax )
												{
													temp2 = Cust[ temp2 ];
													k2++;
												}
												while ( temp2 < N )
												{
													if ( tempPSum + CustPen[ temp2 + Sh ] < PBest )
													{
														if ( tempload - Dem[ temp2 ] <= MaxLoad )
														{
															PBest = tempPSum + CustPen[ temp2 + Sh ];
															Best[ threadIdx.x ] = tempPSum + CustPen[ temp2 + Sh ];
															Blex[ PMax ] = temp;
															Blex[ PMax + 1 ] = temp2;
															Key[ threadIdx.x ] = threadIdx.x;
															BestPlace = idx;
															BPMax = PMax + 2;
															for ( int i = 0; i < PMax; i++ )
															{
																Blex[ i ] = lex[ i ];
															}
														}
														else if ( PMax + 1 < KMax && tempPSum + CustPen[ temp2 + Sh ] < PBest )
														{
															int k3 = k2;
															temp3 = Cust[ temp2 ];
															while ( temp3 == lex[ k3 ] && k3 < PMax )
															{
																temp3 = Cust[ temp3 ];
																k3++;
															}
															while ( temp3 < N )
															{
																if ( tempPSum + CustPen[ temp2 + Sh ] + CustPen[ temp3 + Sh ] < PBest )
																{
																	if ( tempload - Dem[ temp2 ] - Dem[ temp3 ] <= MaxLoad )
																	{
																		PBest = tempPSum + CustPen[ temp2 + Sh ] + CustPen[ temp3 + Sh ];
																		Best[ threadIdx.x ] = tempPSum + CustPen[ temp2 + Sh ] + CustPen[ temp3 + Sh ];
																		Blex[ 0 ] = temp;
																		Blex[ 1 ] = temp2;
																		Blex[ 2 ] = temp3;
																		Key[ threadIdx.x ] = threadIdx.x;
																		BestPlace = idx;
																		BPMax = PMax + 3;
																	}
																}
																temp3 = Cust[ temp3 ];
																while ( temp3 == lex[ k3 ] && k3 < PMax )
																{
																	temp3 = Cust[ temp3 ];
																	k3++;
																}
															}

														}
													}
													temp2 = Cust[ temp2 ];
													while ( temp2 == lex[ k2 ] && k2 < PMax )
													{
														temp2 = Cust[ temp2 ];
														k2++;
													}
												}
											}
											temp = Cust[ temp ];
											while ( temp == lex[ k1 ] && k1 < PMax )
											{
												temp = Cust[ temp ];
												k1++;
											}
										}
									}
								}
								else if ( Arrival <= Due[ CEP ] && PMax <= KMax )
								{
									int temp1 = CEP;
									temp = Cust[ idx ];
									float Arrival1 = Arrival;
									int h = 1;
									while ( temp < N && Arrival1 < Due[ temp1 ] && MNGain[ h ]>0 )
									{
										int h1 = h;
										lex[ PMax ] = temp;
										float ArrivalB = fmaxf( Arrival1 + Ser + CustDist[ temp1 * ( N + 1 ) + temp ], Red[ temp ] );


										tempPSum = PSum + CustPen[ temp + Sh ];
										if ( tempPSum < PBest )
										{

											ArrivalB = fmaxf( ArrivalB + Ser + CustDist[ temp * ( N + 1 ) + Cust[ temp ] ], Red[ Cust[ temp ] ] );
											temp = Cust[ temp ];
											h1++;
											float ArrivalA = fmaxf( Arrival1 + Ser + CustDist[ temp1 * ( N + 1 ) + temp ], Red[ temp ] );
											Gain = ArrivalB - ArrivalA;
											if ( Gain >= MNGain[ h1 ] && tempPSum < PBest )
											{
												load = CarLoadCep;
												for ( int i = 0; i < PMax; i++ )
												{
													load -= Dem[ lex[ i ] ];
												}
												load -= Dem[ lex[ PMax ] ];
												if ( load <= MaxLoad )
												{
													for ( int i = 0; i < PMax; i++ )
													{
														Blex[ i ] = lex[ i ];
													}
													Blex[ PMax ] = lex[ PMax ];
													PBest = tempPSum;
													Best[ threadIdx.x ] = tempPSum;
													Key[ threadIdx.x ] = threadIdx.x;
													BestPlace = idx;
													BPMax = PMax + 1;
												}
												else if ( PMax < KMax )
												{
													if ( idx < N )
													{
														temp4 = Cust[ Car[ idx + Sh ] + N ];
													}
													else
													{
														temp4 = Cust[ idx ];
													}
													while ( temp4 < N )
													{
														if ( temp4 != lex[ PMax ] && ( PMax == 0 || temp4 != lex[ 0 ] ) && tempPSum + CustPen[ temp4 + Sh ] < PBest && load - Dem[ temp4 ] <= MaxLoad )
														{
															for ( int i = 0; i < PMax; i++ )
															{
																Blex[ i ] = lex[ i ];
															}
															PBest = tempPSum + CustPen[ temp4 + Sh ];
															Best[ threadIdx.x ] = tempPSum + CustPen[ temp4 + Sh ];
															Blex[ PMax ] = lex[ PMax ];
															Blex[ PMax + 1 ] = temp4;
															Key[ threadIdx.x ] = threadIdx.x;
															BestPlace = idx;
															BPMax = PMax + 2;
														}
														temp4 = Cust[ temp4 ];
													}
												}

											}
											else if ( tempPSum < PBest - 1 && PMax < KMax && temp < N )
											{
												float Arrival2 = Arrival1;
												temp2 = temp1;
												int h2 = h1;
												while ( temp < N && Arrival2 < Due[ temp2 ] && MNGain[ h2 ]>0 )
												{
													lex[ PMax + 1 ] = temp;
													ArrivalB = fmaxf( ArrivalB + Ser + CustDist[ temp * ( N + 1 ) + Cust[ temp ] ], Red[ Cust[ temp ] ] );
													h2++;
													if ( tempPSum + CustPen[ lex[ PMax + 1 ] + Sh ] < PBest )
													{
														ArrivalA = fmaxf( Arrival2 + Ser + CustDist[ temp2 * ( N + 1 ) + Cust[ temp ] ], Red[ Cust[ temp ] ] );
														Gain = ArrivalB - ArrivalA;
														temp = Cust[ temp ];
														if ( Gain >= MNGain[ h2 ] )
														{
															load = CarLoadCep;
															for ( int i = 0; i <= PMax + 1; i++ )
															{
																load -= Dem[ lex[ i ] ];
															}

															if ( load <= MaxLoad )
															{
																for ( int i = 0; i < PMax; i++ )
																{
																	Blex[ i ] = lex[ i ];
																}
																Blex[ PMax ] = lex[ PMax ];
																Blex[ PMax + 1 ] = lex[ PMax + 1 ];
																PBest = tempPSum + CustPen[ lex[ PMax + 1 ] + Sh ];
																Best[ threadIdx.x ] = tempPSum + CustPen[ lex[ PMax + 1 ] + Sh ];
																Key[ threadIdx.x ] = threadIdx.x;
																BestPlace = idx;
																BPMax = PMax + 2;
															}
															else if ( PMax < KMax - 1 )
															{
																if ( idx < N )
																{
																	temp4 = Cust[ Car[ idx + Sh ] + N ];
																}
																else
																{
																	temp4 = Cust[ idx ];
																}
																while ( temp4 < N )
																{
																	if ( temp4 != lex[ 0 ] && temp4 != lex[ 1 ] && tempPSum + CustPen[ lex[ PMax + 1 ] + Sh ] + CustPen[ temp4 + Sh ] < PBest && load - Dem[ temp4 ] <= MaxLoad )
																	{
																		for ( int i = 0; i < PMax; i++ )
																		{
																			Blex[ i ] = lex[ i ];
																		}
																		Blex[ 0 ] = lex[ 0 ];
																		Blex[ 1 ] = lex[ 1 ];
																		Blex[ 2 ] = temp4;
																		PBest = tempPSum + CustPen[ lex[ 1 ] + Sh ] + CustPen[ temp4 + Sh ];
																		Best[ threadIdx.x ] = tempPSum + CustPen[ lex[ 1 ] + Sh ] + CustPen[ temp4 + Sh ];
																		Key[ threadIdx.x ] = threadIdx.x;
																		BestPlace = idx;
																		BPMax = PMax + 3;
																	}
																	temp4 = Cust[ temp4 ];
																}
															}
														}
														else if ( tempPSum + CustPen[ lex[ PMax + 1 ] + Sh ] < PBest - 1 && PMax < KMax - 1 && temp < N )
														{

															int h3 = h2;
															float Arrival3 = Arrival2;
															temp3 = temp2;
															float ArrivalB1 = ArrivalB;
															while ( temp < N && Arrival3 < Due[ temp3 ] && MNGain[ h3 ]>0 )
															{
																lex[ PMax + 2 ] = temp;
																ArrivalB1 = fmaxf( ArrivalB1 + Ser + CustDist[ temp * ( N + 1 ) + Cust[ temp ] ], Red[ Cust[ temp ] ] );
																h3++;
																if ( tempPSum + CustPen[ lex[ PMax + 1 ] + Sh ] + CustPen[ lex[ PMax + 2 ] + Sh ] < PBest && CarLoadCep - Dem[ lex[ 0 ] ] - Dem[ lex[ 1 ] ] - Dem[ lex[ 2 ] ] )
																{
																	ArrivalA = fmaxf( Arrival3 + Ser + CustDist[ temp3 * ( N + 1 ) + Cust[ temp ] ], Red[ Cust[ temp ] ] );


																	Gain = ArrivalB1 - ArrivalA;
																	temp = Cust[ temp ];
																	if ( Gain >= MNGain[ h3 ] )
																	{

																		Blex[ 0 ] = lex[ 0 ];
																		Blex[ 1 ] = lex[ 1 ];
																		Blex[ 2 ] = lex[ 2 ];
																		PBest = tempPSum + CustPen[ lex[ 1 ] + Sh ] + CustPen[ lex[ 2 ] + Sh ];
																		Best[ threadIdx.x ] = tempPSum + CustPen[ lex[ 1 ] + Sh ] + CustPen[ lex[ 2 ] + Sh ];
																		Key[ threadIdx.x ] = threadIdx.x;
																		BestPlace = idx;
																		BPMax = 3;
																	}
																}
																temp = temp3;
																if ( temp3 == CEP )
																{
																	temp3 = Cust[ idx ];
																}
																else
																{
																	temp3 = Cust[ temp3 ];
																}
																while ( temp3 == lex[ PMax ] || temp3 == lex[ PMax + 1 ] )
																{
																	temp3 = Cust[ temp3 ];
																}
																Arrival3 = fmaxf( Arrival3 + Ser + CustDist[ temp * ( N + 1 ) + temp3 ], Red[ temp3 ] );
																temp = Cust[ temp3 ];
															}
														}
													}
													temp = temp2;
													if ( temp2 == CEP )
													{
														temp2 = Cust[ idx ];
													}
													else
													{
														temp2 = Cust[ temp2 ];
													}
													if ( temp2 == lex[ PMax ] )
													{
														temp2 = Cust[ temp2 ];
													}
													Arrival2 = fmaxf( Arrival2 + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
													temp = Cust[ temp2 ];
												}
											}
										}
										if ( temp1 == CEP )
										{
											temp1 = Cust[ idx ];
											Arrival1 = fmaxf( Arrival1 + Ser + CustDist[ CEP * ( N + 1 ) + temp1 ], Red[ temp1 ] );
										}
										else
										{
											temp1 = Cust[ temp1 ];
											Arrival1 = fmaxf( Arrival1 + Ser + CustDist[ PCust[ temp1 + Sh ] * ( N + 1 ) + temp1 ], Red[ temp1 ] );
										}
										temp = Cust[ temp1 ];
										h++;
									}
								}
							}
							if ( CEPGain > 0 )
							{
								possible = false;
								while ( !possible && !Rdy )
								{
									if ( PMax == 0 )
									{
										PMax++;
										if ( idx < N )
										{
											PGain[ 0 ] = CEPGain;
											PPen[ 0 ] = CEPPen;
											lex[ 0 ] = idx;
											PSum = CustPen[ lex[ 0 ] + Sh ];
										}
										else
										{
											Rdy = true;
										}
									}
									else
									{
										if ( Add == true )
										{
											for ( int i = PMax; i >= 1; i-- )
											{
												lex[ i ] = lex[ i - 1 ];
											}
											lex[ 0 ] = PCust[ lex[ 1 ] + Sh ];
											PMax++;
											if ( lex[ 0 ] >= N )
											{
												if ( PMax == 2 )
												{
													Rdy = true;
												}
												else
												{
													lex[ 0 ] = lex[ 1 ];
													PMax = 2;
													Shift = true;
												}
											}
											else
											{
												PSum = PSum + CustPen[ lex[ 0 ] + Sh ];
											}
										}
										if ( Shift == true )
										{
											PSum -= CustPen[ lex[ 0 ] + Sh ];
											lex[ 0 ] = PCust[ lex[ 0 ] + Sh ];
											if ( lex[ 0 ] < N )
											{
												PSum += CustPen[ lex[ 0 ] + Sh ];
											}
											while ( ( lex[ 0 ] >= N || TW[ Cust[ lex[ 0 ] ] + Sh ] - Red[ Cust[ lex[ 0 ] ] ] < PPen[ PMax - 1 ] || TW[ Cust[ lex[ 0 ] ] + Sh ] - Red[ Cust[ lex[ 0 ] ] ] == 0 ) && !Rdy ) //|| TW[PCust[lex[0]]] - Red[PCust[lex[0]]]>0 
											{
												if ( lex[ 0 ] < N )
												{
													PSum -= CustPen[ lex[ 0 ] + Sh ];
												}
												PMax--;
												if ( PMax != 0 )
												{
													lex[ 0 ] = PCust[ lex[ PMax ] + Sh ];
													if ( lex[ 0 ] < N )
													{
														PSum += CustPen[ lex[ 0 ] + Sh ];
													}
													for ( int i = 1; i < PMax; i++ )
													{
														lex[ i ] = lex[ i + 1 ];
													}
												}
												else
												{
													Rdy = true;
												}
											}

										}
									}
									if ( !Rdy )
									{
										if ( PSum < PBest )
										{
											temp = PCust[ lex[ 0 ] + Sh ];
											if ( temp < N )
											{
												Arrival = TW[ temp + Sh ];
											}
											PLex = 0;
											temp2 = Cust[ temp ];
											while ( temp2 == lex[ PLex ] && PLex < PMax )
											{
												temp2 = Cust[ temp2 ];
												PLex++;
											}
											while ( temp2 != Cust[ idx ] )
											{
												if ( temp < N )
												{
													Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
												}
												else
												{
													Arrival = fmaxf( CustDist[ N * ( N + 1 ) + temp2 ], Red[ temp2 ] );
												}
												temp = temp2;
												temp2 = Cust[ temp2 ];
												while ( temp2 == lex[ PLex ] && PLex < PMax )
												{
													temp2 = Cust[ temp2 ];
													PLex++;
												}
											}
											if ( temp < N )
											{
												Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + CEP ], Red[ CEP ] );
											}
											else
											{
												Arrival = fmaxf( CustDist[ N * ( N + 1 ) + CEP ], Red[ CEP ] );
											}
											if ( CepTW - Arrival < PGain[ PMax - 1 ] )
											{
												if ( Arrival < Due[ CEP ] )
												{
													possible = true;
													Gain = CepTW - Arrival;
												}
												if ( Arrival > Red[ CEP ] && PMax <= KMax && PSum < PBest - 1 )
												{
													Add = true;
													Shift = false;
													PGain[ PMax ] = Arrival - Red[ CEP ];
													PPen[ PMax ] = fmaxf( Arrival - Due[ CEP ], 0.0f );
												}
												else if ( Arrival > Red[ CEP ] )
												{
													Shift = true;
													Add = false;
												}
											}
											else
											{
												possible = true;
												Add = false;
												Shift = true;
												Gain = CepTW - Arrival;
											}
										}
										else
										{
											Shift = true;
											Add = false;
										}
									}
								}

							}
							else
							{
								Rdy = true;
							}

						}
					}
					if ( CEP < 250 && firstround )
					{
						firstround = false;
						if ( threadIdx.x < NCars )
						{
							idx = N + threadIdx.x;
						}
						else
						{
							BehindVehicle = false;
						}
					}
					else
					{
						BehindVehicle = false;
					}

				}
				__syncthreads();
				idx = threadIdx.x;
				//	if (idx < 512){ if (Best[idx] > Best[idx + 512]){ Best[idx] = Best[idx + 512]; Key[idx] = Key[idx + 512]; } } __syncthreads();
				//if (idx < 256){ if (Best[idx] > Best[idx + 256]){ Best[idx] = Best[idx + 256]; Key[idx] = Key[idx + 256]; } } __syncthreads();
				if ( idx < 128 )
				{
					if ( Best[ idx ] > Best[ idx + 128 ] )
					{
						Best[ idx ] = Best[ idx + 128 ]; Key[ idx ] = Key[ idx + 128 ];
					}
				} __syncthreads();
				if ( idx < 64 )
				{
					if ( Best[ idx ] > Best[ idx + 64 ] )
					{
						Best[ idx ] = Best[ idx + 64 ]; Key[ idx ] = Key[ idx + 64 ];
					}
				} __syncthreads();
				if ( idx < 32 )
				{
					if ( Best[ idx ] > Best[ idx + 32 ] )
					{
						Best[ idx ] = Best[ idx + 32 ]; Key[ idx ] = Key[ idx + 32 ];
					}
					if ( Best[ idx ] > Best[ idx + 16 ] )
					{
						Best[ idx ] = Best[ idx + 16 ]; Key[ idx ] = Key[ idx + 16 ];
					}
					if ( Best[ idx ] > Best[ idx + 8 ] )
					{
						Best[ idx ] = Best[ idx + 8 ]; Key[ idx ] = Key[ idx + 8 ];
					}
					if ( Best[ idx ] > Best[ idx + 4 ] )
					{
						Best[ idx ] = Best[ idx + 4 ]; Key[ idx ] = Key[ idx + 4 ];
					}
					if ( Best[ idx ] > Best[ idx + 2 ] )
					{
						Best[ idx ] = Best[ idx + 2 ]; Key[ idx ] = Key[ idx + 2 ];
					}
					if ( Best[ idx ] > Best[ idx + 1 ] )
					{
						Best[ idx ] = Best[ idx + 1 ]; Key[ idx ] = Key[ idx + 1 ];
					}
				}
				__syncthreads();
				if ( Key[ 0 ] == threadIdx.x )
				{

					j++;
					found = true;
					/*******************/
					/*Add Customer CEP*/
					/*****************/
					idx = BestPlace;
					Cust[ CEP ] = Cust[ idx ];
					PCust[ CEP + Sh ] = idx;
					if ( Cust[ idx ] < N )
					{
						PCust[ Cust[ BestPlace ] + Sh ] = CEP;
					}
					if ( idx < N )
					{
						Car[ CEP + Sh ] = Car[ BestPlace + Sh ];
					}
					else
					{
						Car[ CEP + Sh ] = Car[ Cust[ idx ] + Sh ];
					}
					CarLoad[ Car[ CEP + Sh ] + Sh ] += Dem[ CEP ];
					Cust[ idx ] = CEP;
					/*******************/
					/*Delete Customers*/
					/*****************/
					int PLex = 0;
					if ( BPMax == 2 )
					{
						if ( curand( &LocalState ) % ( 2 ) == 1 )
						{
							temp = Blex[ 0 ];
							Blex[ 0 ] = Blex[ 1 ];
							Blex[ 1 ] = temp;
						}
					}
					if ( BPMax == 3 )
					{
						for ( int i = 2; i > 0; i-- )
						{
							int r = curand( &LocalState ) % ( i + 1 );
							if ( r < i )
							{
								temp = Blex[ i ];
								Blex[ i ] = Blex[ r ];
								Blex[ r ] = temp;
							}
						}
					}
					while ( PLex < BPMax )
					{
						temp3 = Blex[ PLex ];
						Cust[ PCust[ temp3 + Sh ] ] = Cust[ temp3 ];
						if ( Cust[ temp3 ] < N )
						{
							PCust[ Cust[ temp3 ] + Sh ] = PCust[ temp3 + Sh ];
						}
						CarLoad[ Car[ temp3 + Sh ] + Sh ] -= Dem[ temp3 ];
						Cust[ temp3 ] = -1;
						EP[ SEP + blockIdx.x * 100 ] = temp3;
						SEP++;
						PLex++;
					}

					/**********************/
					/*Adjust Time windows*/
					/********************/
					if ( idx < N )
					{
						temp = Cust[ Car[ idx + Sh ] + N ];
					}
					else
					{
						temp = Cust[ idx ];
					}
					Arrival = fmaxf( CustDist[ N * ( N + 1 ) + temp ], Red[ temp ] );
					temp2 = temp;
					while ( temp2 != N )
					{
						TW[ temp2 + Sh ] = Arrival;
						if ( TW[ temp2 + Sh ] > Due[ temp2 ] )
						{
							temp3 = Cust[ temp2 ];
							Arrival = fmaxf( TW[ temp + Sh ] + Ser + CustDist[ temp * ( N + 1 ) + temp3 ], Red[ temp3 ] );
							Cust[ PCust[ temp2 + Sh ] ] = Cust[ temp2 ];
							if ( Cust[ temp2 ] < N )
							{
								PCust[ Cust[ temp2 ] + Sh ] = PCust[ temp2 + Sh ];
							}
							CarLoad[ Car[ temp2 + Sh ] + Sh ] -= Dem[ temp2 ];
							Cust[ temp2 ] = -1;
							EP[ SEP + blockIdx.x * 100 ] = temp2;
							SEP++;
							temp2 = temp3;
						}
						else
						{
							temp = temp2;
							temp2 = Cust[ temp2 ];
							Arrival = fmaxf( Arrival + Ser + CustDist[ temp * ( N + 1 ) + temp2 ], Red[ temp2 ] );
						}
					}
				}
				__syncthreads();
			}
{% endhighlight %}
</p>
</details>

I would say phase 3, Penalty for hard customers, is the real heart of this algorithm. I already mentioned the penalty before, but here I am going to explain how it works. If I was not able to insert or squeeze our CEP I increase its penalty for being a hard customer to insert. All customers start with a penalty of 1. Let's assume I was not able to insert customer 1, so I increase the penalty of customer 1 by one(  CustPen[1] = 2)

Now comes the beauty. After giving the CEP a penalty, I am going to try to insert it in all possible places.  I want to find the spot where I can insert the CEP and make it feasible again by removing the customers with the lowest sum of their penalties. It is also possible to remove the CEP itself. 

The removed customers are added to the EP and I go back to the insertion phase. 

This phase feels like a hybrid between machine learning and heuristics. After running this algorithm for a while, the customers which are hard to insert probably get a high penalty value. You will leave those customers inside the solution and build the solution around them.

My comments for this part of the code are that it is challenging to understand what I did. There are so many if statements, so the risk that a warp is cut in half, or even worse is significantly. Also, this code was very error prone, I've spent whole nights fixing bugs in this part.


This was my explanation of the algorithm and how I used my GPU. I got very lucky with breaking one of the best-known solutions, because I didn't come up with a new algorithm. I might have gotten it because I had to make my algorithm based on a paper and maybe I implemented some parts different then how they did it. I will never know, because for a lot of papers they do not publish any source code. Maybe I found it because I solved 100 solution at the same time and was lucky with one of the random seeds. I remember it took about 2 hours to find the new best-known solution, so maybe it would take the CPU to long to find it. In the end it does not matter, because it is my name which is shown on the website :).

Post master thesis

In hindsight I am really happy I did this master thesis. My code might be horrible, but I have learned a lot about memory management, multi-threaded programs and algorithms for vehicle routing problems. Because of those skill I landed a job at ORTEC . At ORTEC I helped developing their vehicle routing solution (https://ortec.com/en/business-areas/routing-loading). The big difference with my toy project is that we actually solve vehicle routing problems for real clients. And those clients do not only have demand and time windows, but many more constraints to take into account. At this job I learned programming properly in c++.

For anyone who has interest doing a similar project as I did, I have some advice:
* Try to parallize only parts of the algoritm. In my case it would probably have been easier if I first implemented it on the CPU and then implement the parts which are slow on the GPU. 
* Use functions (Don't repeat yourself!)
* In the end you must start somewhere, just do it and have fun!

Thanks for reading my blog!!

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
