function maxDist(x, y, X, Y){
    var distmax = 0, dist = 0;
	var c = new Array(2);
	c[0] = X[0]; c[1] = Y[0];
	for (var i = 1; i < X.length; i++){
	    dist = (X[i] - x) * (X[i] - x) + (Y[i] - y) * (Y[i] - y);
		if (dist > distmax){
		    c[0] = X[i];
			c[1] = Y[i];
			distmax = dist;
		}
	}
	return c;
}

function RunKmeans(){
    var ClusterNumber = document.getElementById("ClusterNumber").value;
    if (ClusterNumber > X.length) {
	    ClusterNumber = X.length;
	}
	var ClusterSeedsX = new Array(ClusterNumber);
	var ClusterSeedsY = new Array(ClusterNumber);
	
	document.getElementById("X").innerHTML = X;
	document.getElementById("Y").innerHTML = Y;
    
    var sumX = 0, sumY = 0;
    for (var i = 0; i < X.length; i++){
	    sumX += X[i];
		sumY += Y[i];
	}
	var meanX = sumX/X.length, meanY = sumY/Y.length;
    
	var c = maxDist(meanX, meanY, X, Y);
	ClusterSeedsX[0] = c[0]; ClusterSeedsY[0] = c[1];
	c = maxDist(c[0], c[1], X, Y);
	ClusterSeedsX[1] = c[0]; ClusterSeedsY[1] = c[1];
	
	var dist;
	
	for (var i = 2; i < ClusterNumber; i++){
	    
        ClusterSeedsX[i] = X[0]; ClusterSeedsY[i] = Y[0];
	    maxmindist = 0;
	    for (var j = 0; j < X.length; j++){
		    mindist = (X[j] - ClusterSeedsX[0]) * (X[j] - ClusterSeedsX[0])
			+ (Y[j] - ClusterSeedsY[0]) * (Y[j] - ClusterSeedsY[0]);
			for (var k = 1; k < i; k++){
				dist = (X[j] - ClusterSeedsX[k]) * (X[j] - ClusterSeedsX[k])
				        + (Y[j] - ClusterSeedsY[k]) * (Y[j] - ClusterSeedsY[k]);
		        if (mindist > dist)
					mindist = dist;
	        }
			if (mindist > maxmindist){
			    if ( ( !ClusterSeedsX.includes(X[j]) ) && ( !ClusterSeedsY.includes(Y[j])) ) {
			        ClusterSeedsX[i] = X[j];
			        ClusterSeedsY[i] = Y[j];
     		        maxmindist = mindist;
				}
			}
	    }
	}
	
	var X1 = new Array, Y1 = new Array;
	var maxInt, val, maxHex = "ffffff";
	maxInt = parseInt(maxHex, 16);
	document.getElementById("X").innerHTML = ClusterSeedsX;
	document.getElementById("Y").innerHTML = ClusterSeedsY;
	color = ["blue","red","green","yellow", "black", "brown", "cyan", "#FF00FF", "#808080", "RosyBrown"];
	sign = ["circle", "square"]
	ClearCanvas();
	for (var i = 0; i < ClusterNumber; i++)
	    drawCoordinates(ClusterSeedsX[i], ClusterSeedsY[i], color[i], 5, sign[i % 2]);	
	
	var Clustered = new Array, cmeansX = new Array, cmeansY = new Array;
	
	for (var i = 0; i < X.length; i++)
	    Clustered.push(-1);
	
	var swap = 1, n = 0;
    while ((n < 100) & (swap == 1)){
        n = n + 1;
        swap = 0;
            
        for (var i = 0; i < X.length; i++){
		    
			for (var j = 0; j < ClusterNumber; j++){
                cmeanX[j] = ClusterSeedsX[j];
				cmeanY[j] = ClusterSeedsY[j];
			}
            for (var j = 0; j < Clusternumber; j++){
                if (Clustered[i] == j){
                    cmeanX[j] = ClustermeansX[j];
					cmeanY[j] = ClustermeansY[j];
				}
                else:
                    cmean[j] = [( Clustermeans[j][k] * Clusterlength[j] \
                                        + data[i][k]) / (Clusterlength[j] + 1)\
                                    for k in range( K ) ]
			}
                    
            ci = 0
            distc0 = math.sqrt( sum([(data[i][j] - cmean[0][j])**2 \
                            for j in range(K)]) )
            for c in range(1, Clusternumber):
                distcj = math.sqrt( sum( [ (data[i][j] - cmean[c][j])**2 \
                            for j in range( K ) ] ) )
                if distcj < distc0:
                    ci = c
                    distc0 = distcj

            if ci != Clustered[i]:
                swap = 1
                    
            if (Clustered[i] == -1):
                Clustered[i] = ci
                Clusterlength[ci] = Clusterlength[ci] + 1
                Clustermeans[ci] = [cmean[ci][j] for j in range(K)]
            else:
                if (ci != Clustered[i]):
                    #print(ci)
                    Clustermeans[Clustered[i]] \
                        = [(Clustermeans[Clustered[i]][j] \
                            * Clusterlength[Clustered[i]] \
                          - data[i][j]) / (Clusterlength[Clustered[i]] - 1) \
                           for j in range(len(data[i])) ]
                    #print(ci)
                    Clustermeans[ci] = [cmean[ci][j] for j in range(len(cmean[ci]))]
                    Clusterlength[Clustered[i]] = Clusterlength[Clustered[i]] - 1
                    Clusterlength[ci] = Clusterlength[ci] + 1
                    Clustered[i] = ci
	    
	
    return 0;
}