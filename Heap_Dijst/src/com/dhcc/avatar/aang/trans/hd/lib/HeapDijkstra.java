package com.dhcc.avatar.aang.trans.hd.lib;

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.Vector;

/**
 * Heap + Dijkstra算法求单源最短路径 采用邻接表存储图数据； 邻接表结构: 头结点--Node对象数组；
 * 邻接节点--Node对象中的HashMap；
 * 输入： 
 *   int[] counts = new int[] { 5, 7 };
	 int[][] linknodes = new int[][] { { 0, 0, 0, 2, 2, 3, 4 }, 
	 { 1, 2, 4, 1, 3, 1, 3 }, { 100, 30, 10, 60, 60, 10, 50 } };
 * 返回结果：Map<Vector<Integer>, Integer>
 * {[4, 3]=60, [4, 3, 1]=70, [2]=30, [4]=10}111
 * 
 */

public class HeapDijkstra {// 11111
	public HeapDijkstra(int[] counts, int[][] linknodes) {
		nodeCount = counts[0];
		edgeCount = counts[1];
		firstArray = new Node[nodeCount + 1];
		for (int i = 0; i < nodeCount; i++) {
			firstArray[i] = new Node();
		}
		for (int i = 0; i < linknodes[0].length; i++) {
			int begin = linknodes[0][i];
			int end = linknodes[1][i];
			int edge = linknodes[2][i];
			firstArray[begin].addEdge(end, edge);
		}
	}

	private int nodeCount;
	private int edgeCount;
	// 邻接表表头数组
	private Node[] firstArray;
	// 最短路径数组
	private int[][] dist;
	private int[] ref;
	private int max = 1000000;
	private Vector<Integer>[] path;

	/**
	 * Node Class
	 * 
	 */
	private class Node {
		// 邻接顶点map
		private HashMap<Integer, Integer> map = null;

		public void addEdge(int end, int edge) {
			if (this.map == null) {
				this.map = new HashMap<Integer, Integer>();
			}
			this.map.put(end, edge);
		}
	}

	/**
	 * Heap + Dijkstra Algorithm
	 * 
	 * @param counts
	 * @param linknodes
	 */
	private Map<Vector<Integer>, Integer> djst() {
		Map<Vector<Integer>, Integer> result = new HashMap<>();
		dist = new int[2][nodeCount];
		ref = new int[nodeCount];
		path = new Vector[nodeCount];
		for (int i = 0; i < nodeCount; i++) {
			path[i] = new Vector<Integer>();
		}

		Node tempNode = firstArray[0];
		for (int i = 1; i < nodeCount; i++) {
			HashMap<Integer, Integer> tempMap = tempNode.map;
			dist[0][i] = tempMap.containsKey(i) ? tempMap.get(i) : max;
			dist[1][i] = i;
			ref[i] = i;
			if (tempMap.containsKey(i))
				path[i].add(i);
			minUp(i);
		}
		int flag = nodeCount - 1;
		while (flag >= 1) {
			int index = dist[1][1];
			changeKey(1, flag);
			maxDown(1, --flag);
			// 用indx这个点去更新它的邻接点到开始点的距离
			HashMap<Integer, Integer> m = firstArray[index].map;
			if (m == null) {
				continue;
			}
			Set<Integer> set = m.keySet();
			Iterator<Integer> it = set.iterator();
			while (it.hasNext()) {
				int num = it.next();
				if (m.get(num) + dist[0][flag + 1] < dist[0][ref[num]]) {
					dist[0][ref[num]] = m.get(num) + dist[0][flag + 1];
					Vector<Integer> temp;
					temp = (Vector<Integer>) path[dist[1][flag + 1]].clone();
					temp.add(num);
					path[num].clear();
					path[num] = temp;
					minUp(ref[num]);
				}
			}
		}
		for (int i = 1; i < nodeCount; i++) {
			result.put(path[i], dist[0][ref[i]]);
		}
		return result;
	}

	/**
	 * 最大值下沉
	 * 
	 * @param index
	 * @param end
	 */
	private void maxDown(int index, int end) {
		int temp = dist[0][index];
		int left = index * 2 - 1;
		while (left < end) {
			// 判断左右子节点大小
			if (left + 1 <= end && dist[0][left + 1] < dist[0][left]) {
				left++;
			}
			// 如果左右子节点都比temp大的话结束下沉操作
			if (dist[0][left] > temp) {
				break;
			}
			// 交换子节点和父节点
			changeKey(index, left);
			index = left;
			left = index * 2 - 1;
		}
	}

	/**
	 * 小值上升
	 * 
	 * @param n
	 */
	private void minUp(int n) {
		int f = (n + 1) / 2;
		while (f >= 1 && dist[0][f] > dist[0][n]) {
			changeKey(f, n);
			n = f;
			f = (n + 1) / 2;
		}
	}

	/**
	 * change two values in array
	 * 
	 * @param a
	 * @param b
	 */
	private void changeKey(int a, int b) {
		int n = dist[1][a];
		int m = dist[1][b];
		int temp = ref[n];
		ref[n] = ref[m];
		ref[m] = temp;
		temp = dist[0][a];
		dist[0][a] = dist[0][b];
		dist[0][b] = temp;
		temp = dist[1][a];
		dist[1][a] = dist[1][b];
		dist[1][b] = temp;
	}

	/**
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {
		int[] counts = new int[] { 5, 7 };
		int[][] linknodes = new int[][] { { 0, 0, 0, 2, 2, 3, 4 }, { 1, 2, 4, 1, 3, 1, 3 },
				{ 100, 30, 10, 60, 60, 10, 50 } };
		HeapDijkstra heapDijkstra = new HeapDijkstra(counts, linknodes);
		Map<Vector<Integer>, Integer> result = heapDijkstra.djst();
		System.out.println(result);
	}
}
