package jian_zhi_offer;


import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Stack;

/**
 * 剑指offer
 * 来源：力扣（LeetCode）
 * 链接：https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof
 *
 * @author boomzy
 * @date 2021/12/14 22:43
 */
public class Main {

    /**
     * 剑指 Offer 03. 数组中重复的数字
     * <p>
     * 在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，
     * 但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字
     * <p>
     * eg:
     * 输入：
     * [2, 3, 1, 0, 2, 5, 3]
     * 输出：2 或 3
     *
     * @param nums
     * @return
     */
    public int findRepeatNumber(int[] nums) {
        /*
            时间复杂度为O(N)，空间复杂度为O(1)的算法：
            从头到尾依次扫描这个数组中的数字，当扫描到下标为i的数字时，首先比较这个数字(记为m)是否等于i。如果是则接着扫描
            下一个数字；如果不是，则再拿这个数和第m个数字进行比较。如果它和第m个数字相等，则就找到了一个重复数字；如果不相等
            则将第i个和第m个数字交换，接下来继续重复比较并交换的过程即可

            eg:
            nums=[2, 3, 1, 0, 2, 5, 3]
            > nums[0]=2记为m, 不等于i=0，将m=2和nums[m]=1比较，不相等，交换。交换后数组[1, 3, 2, 0, 2, 5, 3]
            > nums[0]=1记为m, 不等于i=0，将m=1和nums[m]=3比较，不相等，交换。交换后数组[3, 1, 2, 0, 2, 5, 3]
            > nums[0]=3记为m, 不等于i=0，将m=3和nums[m]=0比较，不相等，交换。交换后数组[0, 1, 2, 3, 2, 5, 3]
            > nums[0]=0记为m, 等于i=0，继续扫描下一个数字，接下来几个数字均相等
            ...
            > nums[4]=2记为m, 不等于i=0，将m=2和nums[m]=2比较，相等，说明为重复数字，结束。

            每个数字最多只会交换两次就能找到正确位置，即时间复杂度为O(N)，空间复杂度为O(1)

         */
        if (nums == null || nums.length == 0) {
            return 0;
        }

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < 0 || nums[i] > nums.length - 1) {
                return 0;
            }
        }

        for (int i = 0; i < nums.length; i++) {
            while (i != nums[i]) {
                if (nums[i] == nums[nums[i]]) {
                    return nums[i];
                }
                // swap
                int temp = nums[i];
                nums[i] = nums[temp];
                nums[temp] = temp;
            }
        }

        return 0;
    }

    /**
     * 剑指 Offer 04. 二维数组中的查找
     * <p>
     * 在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。
     * 请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
     * <p>
     * 示例:
     * 现有矩阵 matrix 如下：
     * <p>
     * [
     * [1,   4,  7, 11, 15],
     * [2,   5,  8, 12, 19],
     * [3,   6,  9, 16, 22],
     * [10, 13, 14, 17, 24],
     * [18, 21, 23, 26, 30]
     * ]
     * 给定 target = 5，返回 true。
     * 给定 target = 20，返回 false。
     *
     * @param matrix
     * @param target
     * @return
     */
    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        /*
            从右上角开始，如果该数字为要查找的数字，返回true
            如果该数字大于要查找的数字，则剔除该数字所在的列，即column--
            如果该数字小于等于要查找的数字，则剔除该数字所在的行，row++
         */
        if (matrix == null || matrix.length == 0) {
            return false;
        }
        // begin index
        int row = 0;
        int column = matrix[0].length - 1;
        while (row < matrix.length && column >= 0) {
            if (matrix[row][column] == target) {
                return true;
            }
            if (matrix[row][column] > target) {
                column--;
            } else {
                row++;
            }
        }
        return false;
    }

    /**
     * 剑指 Offer 05. 替换空格
     * <p>
     * 请实现一个函数，把字符串 s 中的每个空格替换成"%20"。
     * <p>
     * 示例 1：
     * 输入：s = "We are happy."
     * 输出："We%20are%20happy."
     * <p>
     * 限制：
     * 0 <= s 的长度 <= 10000
     *
     * @param s
     * @return
     */
    public String replaceSpace(String s) {
        if (s == null || "".equals(s)) {
            return "";
        }

        int length = s.length();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == ' ') {
                sb.append("%20");
            } else {
                sb.append(s.charAt(i));
            }
        }
        return sb.toString();
    }

    /**
     * 剑指 Offer 06. 从尾到头打印链表
     * <p>
     * 输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。
     *
     * @param head
     * @return
     */
    public int[] reversePrint(ListNode head) {
        int count = 0;
        ListNode curr = head;
        while (curr != null) {
            curr = curr.next;
            count++;
        }
        int[] res = new int[count];
        for (int i = res.length - 1; i >= 0; i--) {
            res[i] = head.val;
            head = head.next;
        }
        return res;
    }

    /**
     * 剑指 Offer 07. 重建二叉树
     * <p>
     * 输入某二叉树的前序遍历和中序遍历的结果，请构建该二叉树并返回其根节点。
     * 假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
     * <p>
     * eg:
     * Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
     * Output: [3,9,20,null,null,15,7]
     * <p>
     * 限制：
     * 0 <= 节点个数 <= 5000
     *
     * @param preOrder
     * @param inorder
     * @return
     */
    public TreeNode buildTree(int[] preOrder, int[] inorder) {
        if (preOrder == null || preOrder.length == 0) {
            return null;
        }

        if (preOrder.length > 5000) {
            return null;
        }

        TreeNode root = new TreeNode(preOrder[0]);

        int index = findRootIndex(preOrder, inorder);

        root.left = buildTree(Arrays.copyOfRange(preOrder, 1, index + 1),
                Arrays.copyOfRange(inorder, 0, index));
        root.right = buildTree(Arrays.copyOfRange(preOrder, index + 1, preOrder.length),
                Arrays.copyOfRange(inorder, index + 1, inorder.length));

        return root;
    }

    private int findRootIndex(int[] preOrder, int[] inorder) {
        for (int i = 0; i < inorder.length; i++) {
            if (inorder[i] == preOrder[0]) {
                return i;
            }
        }
        return 0;
    }

    /**
     * 剑指 Offer 08. 二叉树的下一个结点
     * <p>
     * 给定一个二叉树其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。
     * 注意，树中的结点不仅包含左右子结点，同时包含指向父结点的next指针
     * <p>
     * 示例:
     * 输入:{8,6,10,5,7,9,11},8
     * 返回:9
     * 解析:这个组装传入的子树根节点，其实就是整颗树，中序遍历{5,6,7,8,9,10,11}，
     * 根节点8的下一个节点就是9，应该返回{9,10,11}，后台只打印子树的下一个节点，所以只会打印9
     *
     * @param pNode
     * @return
     */
    public TreeLinkNode GetNext(TreeLinkNode pNode) {
        if (pNode == null) {
            return null;
        }
        /*
            两种情况：
            1.如果一个节点有右子树，由于是中序遍历顺序，那么它的下一个节点就是它的右子树中的左子节点。
            即从右子节点出发一直沿着指向左子树的指针寻找
            2.如果一个节点没有右子树，如果节点是它父节点的左子节点，那么它的下一个节点就是父节点
            3.如果一个节点既没有右子树，也是它父节点的右子节点，需要沿着父节点指针一直遍历，直到找到一个是它父节点的
            左子节点的节点。
         */
        TreeLinkNode pNext = null;
        // 情况一
        if (pNode.right != null) {
            TreeLinkNode pRight = pNode.right;
            while (pRight.left != null) {
                pRight = pRight.left;
            }
            pNext = pRight;
            return pNext;
        }

        // 情况二、三合起来写，因为二算是三的简易情况
        if (pNode.next != null) {
            TreeLinkNode current = pNode;
            TreeLinkNode parent = pNode.next;
            while (parent != null && current == parent.right) {
                current = parent;
                parent = parent.next;
            }
            pNext = parent;
        }

        return pNext;
    }

    /**
     * 剑指 Offer 10. 斐波那契数列
     * F(0) = 0,   F(1) = 1
     * F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
     * <p>
     * 答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。
     *
     * @param n
     * @return
     */
    public int fib(int n) {
        if (n <= 1) {
            return n;
        }
        final int MOD = 1000000007;
        int fibN = 0;
        int minusOne = 1; // 前一个数，即f(1)
        int minusTwo = 0; // 前两个数，即f(0)
        // 从f(2)开始
        for (int i = 2; i <= n; i++) {
            fibN = (minusTwo + minusOne) % MOD;
            minusTwo = minusOne;
            minusOne = fibN;
        }
        return fibN;
    }

    /**
     * 剑指 Offer 11. 旋转数组的最小数字
     * <p>
     * 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
     * <p>
     * 给你一个可能存在 重复 元素值的数组 numbers ，它原来是一个升序排列的数组，并按上述情形进行了一次旋转。
     * 请返回旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一次旋转，该数组的最小值为1。  
     * <p>
     * 示例 1：
     * 输入：[3,4,5,1,2]
     * 输出：1
     * <p>
     * 示例 2：
     * 输入：[2,2,2,0,1]
     * 输出：0
     *
     * @param numbers
     * @return
     */
    public int minArray(int[] numbers) {
        /*
            问题思路：
            可以注意到旋转之后的数组实际上划分为两个递增子数组，前面子数组的子元素都大于后面子数组的子元素。则考虑使用二分
            1.使用二分，设定头指针指向第一个元素，尾指针指向最后一个元素。
            2.首先找到数组中间元素，若该元素位于第一个递增子数组，则它应该大于或等于头指针指向的元素，要找到的目标元素应该在
            该中间元素的后面，则修改头指针的位置到中间元素位置，缩小范围。
            若该元素位于第二个递增子数组则同理，它应该小于或等于尾指针指向的元素，要找到的目标元素应该在
            该中间元素的前面，则修改尾指针的位置到中间元素位置，缩小范围。
            3.最终头指针将指向前面子数组的最后一个元素，尾指针将指向后面子数组的第一个元素，即两个指针最后会相邻，而尾指针
            指向的将是最终的结果元素。
            4.此处存在特例，若头指针和尾指针和中间指针三个的元素值一致，则无法通过上述实现，只能遍历数组找到目标元素
         */
        if (numbers == null || numbers.length == 0) {
            return 0;
        }

        int leftIndex = 0;
        int rightIndex = numbers.length - 1;
        int midIndex = leftIndex;
        while (numbers[leftIndex] >= numbers[rightIndex]) {
            // 步骤3，终止的判断
            if (rightIndex - leftIndex == 1) {
                return numbers[rightIndex];
            }

            midIndex = leftIndex + (rightIndex - leftIndex) / 2;
            // 步骤4，特殊情况判断
            if (numbers[leftIndex] == numbers[rightIndex] && numbers[leftIndex] == numbers[midIndex]) {
                Arrays.sort(numbers);
                return numbers[0];
            }

            // 步骤1和2的循环判断
            if (numbers[midIndex] >= numbers[leftIndex]) {
                leftIndex = midIndex;
            } else if (numbers[midIndex] <= numbers[rightIndex]) {
                rightIndex = midIndex;
            }
        }

        return numbers[midIndex];
    }

    /**
     * 剑指 Offer 12. 矩阵中的路径
     * <p>
     * 给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；
     * 否则，返回 false 。
     * <p>
     * 单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。
     * 同一个单元格内的字母不允许被重复使用。
     * <p>
     * 示例 1：
     * 输入：board =
     * [
     * ["A","B","C","E"],
     * ["S","F","C","S"],
     * ["A","D","E","E"]
     * ], word = "ABCCED"
     * 输出：true
     * <p>
     * 示例 2：
     * 输入：board =
     * [
     * ["a","b"],
     * ["c","d"]
     * ], word = "abcd"
     * 输出：false
     *
     * @param board 矩阵
     * @param word  要匹配的字符串
     * @return
     */
    public boolean exist(char[][] board, String word) {
        // 回溯：dfs + 剪枝
        if (board == null || board.length == 0 || board[0].length == 0 || word == null || "".equals(word)) {
            return false;
        }

        boolean[][] visited = new boolean[board.length][board[0].length];

        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (solve(board, word, i, j, visited, 0)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * 第10题帮助函数
     *
     * @param board   矩阵
     * @param row     当前行
     * @param column  当前列
     * @param visited 遍历过的标志
     * @param index   下标，用于记录已经遍历了目标字符串的第几个字符了
     * @return
     */
    private boolean solve(char[][] board, String word, int row, int column, boolean[][] visited, int index) {
        // 退出条件，越界判断
        if (row < 0 || row > board.length - 1 || column < 0 || column > board[0].length - 1 || visited[row][column]) {
            return false;
        }

        // 退出条件，匹配到某一位置不满足条件
        if (board[row][column] != word.charAt(index)) {
            return false;
        }

        // 退出条件，已匹配成功
        if (index == word.length() - 1) {
            return true;
        }

        // 匹配过程，匹配到
        visited[row][column] = true;
        boolean flag = solve(board, word, row + 1, column, visited, index + 1) ||
                solve(board, word, row - 1, column, visited, index + 1) ||
                solve(board, word, row, column + 1, visited, index + 1) ||
                solve(board, word, row, column - 1, visited, index + 1);
        visited[row][column] = false;

        return flag;
    }

    private int sum = 0;

    /**
     * 剑指 Offer 13. 机器人的运动范围
     * <p>
     * 地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，
     * 它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。
     * 例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，
     * 因为3+5+3+8=19。请问该机器人能够到达多少个格子？
     *
     * @param m
     * @param n
     * @param k
     * @return
     */
    public int movingCount(int m, int n, int k) {
        if (m <= 0 || n <= 0) {
            return 0;
        }

        boolean[][] visited = new boolean[m][n];
        solve(0, 0, m, n, visited, k);
        return sum;
    }

    private void solve(int row, int col, int rows, int cols, boolean[][] visited, int k) {
        if (row < 0 || col < 0 || row > rows - 1 || col > cols - 1 || visited[row][col] || (getDigitSum(row) + getDigitSum(col) > k)) {
            return;
        }

        // 无需进行回溯，因为只需要判断该点能否抵达
        visited[row][col] = true;
        sum++;
        solve(row + 1, col, rows, cols, visited, k);
        solve(row - 1, col, rows, cols, visited, k);
        solve(row, col + 1, rows, cols, visited, k);
        solve(row, col - 1, rows, cols, visited, k);
    }

    /**
     * 计算x各位之和
     *
     * @param x
     * @return
     */
    private int getDigitSum(int x) {
        int res = 0;
        while (x != 0) {
            res += x % 10;
            x /= 10;
        }
        return res;
    }

    /**
     * 剑指 Offer 14-I. 剪绳子
     * <p>
     * 给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），
     * 每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？
     * 例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。
     * <p>
     * 示例 1：
     * 输入: 2
     * 输出: 1
     * 解释: 2 = 1 + 1, 1 × 1 = 1
     * <p>
     * 示例 2:
     * 输入: 10
     * 输出: 36
     * 解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36
     * <p>
     * 2 <= n <= 58
     *
     * @param n
     * @return
     */
    public int cuttingRope(int n) {
        /*
            动态规划
            定义f(n)为将长度为n的绳子剪成若干段后各段长度乘积的最大值
            f(n) = max(f(n)*f(n-i)) 其中0<i<n
         */
        if (n < 2) {
            return 0;
        }
        if (n == 2) {
            return 1;
        }
        // 1 * 2的情况最大
        if (n == 3) {
            return 2;
        }
        // dp[i]表示的意思其实是对于长度为n的绳子(n > i)，长度为i的绳子所能做出的贡献最大是多少
        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = 1;
        dp[2] = 2;
        // dp[i]作为一个最大的子绳子长度，去跟后面相乘肯定为最大
        dp[3] = 3;
        for (int k = 1; k <= n; k++) {
            for (int i = 1; i <= k / 2; i++) { // 乘法结合律：a*b=b*a，只需要乘一半即可
                dp[k] = Math.max(dp[k], dp[k - i] * dp[i]);
            }
        }
        return dp[n];
    }

    /**
     * 剑指 Offer 14- II. 剪绳子 II
     * <p>
     * 题目同I，答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。
     * <p>
     * 2 <= n <= 1000
     *
     * @param n
     * @return
     */
    public int cuttingRope2(int n) {
        BigInteger[] dp = new BigInteger[n + 1];
        if (n < 2) {
            return 0;
        }
        if (n == 2) {
            return 1;
        }
        // 1 * 2的情况最大
        if (n == 3) {
            return 2;
        }

        dp[1] = BigInteger.ONE;
        dp[2] = BigInteger.valueOf(2);
        dp[3] = BigInteger.valueOf(3);
        for (int k = 4; k <= n; k++) {
            dp[k] = BigInteger.ZERO;
            for (int i = 1; i <= k / 2; i++) {
                BigInteger temp = dp[i].multiply(dp[k - i]);
                if (dp[k].compareTo(temp) < 0) {
                    dp[k] = temp;
                }
            }
        }
        return dp[n].mod(BigInteger.valueOf(1000000007)).intValue();
    }

    /**
     * 剑指 Offer 15. 二进制中1的个数
     * <p>
     * 编写一个函数，输入是一个无符号整数（以二进制串的形式），
     * 返回其二进制表达式中数字位数为 '1' 的个数（也被称为 汉明重量).）
     * <p>
     * eg:
     * 输入：n = 11 (控制台输入 00000000000000000000000000001011)
     * 输出：3
     * 解释：输入的二进制串 00000000000000000000000000001011 中，共有三位为 '1'。
     *
     * @param n
     * @return
     */
    public int hammingWeight(int n) {
        /*
            将一个整数减去1，然后和原整数做与运算，会将该整数最右边的1变成0。（举例，二进制1100减去1变为1011）
            那么一个整数的二进制表示中，有多少个1，就可以进行多少次这样的操作。
         */

        int count = 0;
        while (n != 0) {
            n = (n - 1) & n;
            count++;
        }
        return count;
    }

    /**
     * 剑指 Offer 16. 数值的整数次方
     * 实现 pow(x, n) ，即计算 x 的 n 次幂函数（即，xn）。不得使用库函数，同时不需要考虑大数问题。
     * <p>
     * 示例 1：
     * 输入：x = 2.00000, n = 10
     * 输出：1024.00000
     * <p>
     * 示例 2：
     * 输入：x = 2.10000, n = 3
     * 输出：9.26100
     * <p>
     * 示例 3：
     * 输入：x = 2.00000, n = -2
     * 输出：0.25000
     * 解释：2-2 = 1/22 = 1/4 = 0.25
     *
     * @param x
     * @param n
     * @return
     */
    public double myPow(double x, int n) {
        /*
            递归求解，求32次方时已经知道了16次方，求16次方又已经知道了8次方，以此类推
         */
        if (n == 0) {
            return 1;
        }
        if (n == 1) {
            return x;
        }
        if (n == -1) {
            return 1 / x;
        }
        double res = myPow(x, n / 2);
        res *= res;
        if (n % 2 == 1) {
            // 正奇数
            res *= x;
        } else if (n % 2 == -1) {
            // 负奇数
            res *= 1 / x;
        }
        return res;
    }

    /**
     * 剑指 Offer 17. 打印从1到最大的n位数
     * <p>
     * 输入数字 n，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3
     * 一直到最大的 3 位数 999。
     * <p>
     * 示例 1:
     * 输入: n = 1
     * 输出: [1,2,3,4,5,6,7,8,9]
     *
     * @param n
     * @return
     */
    public int[] printNumbers(int n) {
        /*
            思路一：最终结果的数量为Math.pow(10,n)-1，例如n=2, 则最大的结果是99
            只需要得到Math.pow(10,n)-1作为结果长度，从1开始遍历即可。但存在int溢出的风险

            思路二：考虑大数，用字符串输出 回溯，用字符串的方式求n位数之内的全排列，
            回溯的时候需要遍历到每个数字，且需要一个列表保存每次dfs触底生成的数字，
            所以时空复杂度均为O(10^n)
         */
        if (n <= 0) {
            return null;
        }

        List<Character> path = new ArrayList<>();
        List<String> ansList = new ArrayList<>();
        dfs(n, 0, ansList, path);
        int[] res = new int[ansList.size()];
        for (int i = 0; i < res.length; i++) {
            res[i] = Integer.parseInt(ansList.get(i));
        }
        return res;
    }

    private void dfs(int n, int depth, List<String> ansList, List<Character> path) {
        // 构建字符串形式的数字
        if (depth == n) {
            StringBuilder sb = new StringBuilder();
            boolean flag = false;

            for (int i = 0; i < n; i++) {
                Character character = path.get(i);
                // 忽略字符串中的前导0字符
                if (flag || !character.equals('0')) {
                    flag = true;
                    sb.append(character);
                }
            }

            // 全是由0组成，跳过
            if (!flag) {
                return;
            }

            ansList.add(sb.toString());
            return;
        }

        for (int i = 0; i < 10; i++) {
            // 当前路径中添加当前数字的字符形式
            path.add(String.valueOf(i).charAt(0));
            dfs(n, depth + 1, ansList, path);
            path.remove(path.size() - 1);
        }
    }

    /**
     * 剑指 Offer 18. 删除链表的节点
     * <p>
     * 给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。
     * 返回删除后的链表的头节点。
     * <p>
     * 注意：此题对比原题有改动
     * <p>
     * 示例 1:
     * 输入: head = [4,5,1,9], val = 5
     * 输出: [4,1,9]
     * 解释: 给定你链表中值为 5 的第二个节点，那么在调用了你的函数之后，该链表应变为 4 -> 1 -> 9.
     * <p>
     * 示例 2:
     * 输入: head = [4,5,1,9], val = 1
     * 输出: [4,5,9]
     * 解释: 给定你链表中值为 1 的第三个节点，那么在调用了你的函数之后，该链表应变为 4 -> 5 -> 9
     *
     * @param head
     * @param val
     * @return
     */
    public ListNode deleteNode(ListNode head, int val) {
        /*
            单指针
                         cur cur.next
                         |   |
            dummy->4->5->1-> 9 ->null
            |      |
            cur   cur.next

            cur=dummy,cur.next就是head。判断cur.next.val是否等于Val，是则删除该节点
            边界条件：
            左边界cur.next != null
            右边界:由图可知，若cur.next.val在链表最后一个，即图中的9。那么需要将cur指向cur.next的null
            并且跳出

         */
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode cur = dummy;
        while (cur.next != null) {
            if (cur.next.val == val) {
                // 右边界:当val是最后一个节点
                if (cur.next.next == null) {
                    cur.next = null;
                    break;
                }

                // 删除
                cur.next = cur.next.next;
            }
            cur = cur.next;
        }
        return dummy.next;
    }

    /**
     * 剑指 Offer 20. 表示数值的字符串
     * <p>
     * 请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。
     * <p>
     * 数值（按顺序）可以分成以下几个部分：
     * 若干空格
     * 一个 小数 或者 整数
     * （可选）一个 'e' 或 'E' ，后面跟着一个 整数
     * 若干空格
     * <p>
     * 小数（按顺序）可以分成以下几个部分：
     * （可选）一个符号字符（'+' 或 '-'）
     * 下述格式之一：
     * 至少一位数字，后面跟着一个点 '.'
     * 至少一位数字，后面跟着一个点 '.' ，后面再跟着至少一位数字
     * 一个点 '.' ，后面跟着至少一位数字
     * <p>
     * 整数（按顺序）可以分成以下几个部分：
     * （可选）一个符号字符（'+' 或 '-'）
     * 至少一位数字
     * 部分数值列举如下：
     * ["+100", "5e2", "-123", "3.1416", "-1E-16", "0123"]
     * 部分非数值列举如下：
     * ["12e", "1a3.14", "1.2.3", "+-5", "12e+5.4"]
     *  
     *
     * @param s
     * @return
     */
    public boolean isNumber(String s) {
        /*
            要满足以下几个条件：
            .之前不能有e/E
            e之前不能有e，且必须有数字，eg:123e10，并且不能为12e这种格式
            +/-符号 要出现在第一个位置，或e后面的第一个位置
         */

        if (s == null || s.length() == 0) {
            return false;
        }

        // 表示num是否出现
        boolean isNum = false;
        // 表示.是否出现
        boolean isDot = false;
        // 表示e/E是否出现
        boolean isE = false;

        char[] arr = s.trim().toCharArray();
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] >= '0' && arr[i] <= '9') {
                isNum = true;
            } else if (arr[i] == '.') {
                // .之前不能有e/E
                if (isDot || isE) {
                    return false;
                }
                isDot = true;
            } else if (arr[i] == 'e' || arr[i] == 'E') {
                // e之前不能有e，且必须有数字，eg:123e10，
                if (isE || !isNum) {
                    return false;
                }
                isE = true;
                // 并且不能为12e，12e+这种格式
                // 重置isNum，因为'e'或'E'之后也必须接上整数，防止出现 123e或者123e+的非法情况
                isNum = false;
            } else if (arr[i] == '+' || arr[i] == '-') {
                // +/-符号 要出现在第一个位置，或e后面的第一个位置
                if (i != 0 && arr[i - 1] != 'e' && arr[i - 1] != 'E') {
                    return false;
                }
            } else {
                // 其他字符情况
                return false;
            }
        }

        return isNum;
    }
}

/**
 * 剑指 Offer 09. 用两个栈实现队列
 * <p>
 * 用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，
 * 分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )
 */
class CQueue {

    private Stack<Integer> pushStack;

    private Stack<Integer> popStack;

    public CQueue() {
        pushStack = new Stack<>();
        popStack = new Stack<>();
    }

    // 添加元素：直接往插入栈中插入即可
    public void appendTail(int value) {
        pushStack.push(value);
    }

    /**
     * 删除元素：
     * 当弹出栈不为空时，直接弹出栈顶。
     * 当弹出栈为空时，将插入栈的元素不断入栈到弹出栈中。最后返回弹出栈的栈顶
     *
     * @return
     */
    public int deleteHead() {
        if (!popStack.isEmpty()) {
            return popStack.pop();
        }
        while (!pushStack.isEmpty()) {
            popStack.push(pushStack.pop());
        }
        return popStack.isEmpty() ? -1 : popStack.pop();
    }
}

class ListNode {
    int val;
    ListNode next;

    ListNode(int x) {
        val = x;
    }
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode(int x) {
        val = x;
    }
}

class TreeLinkNode {
    int val;
    TreeLinkNode left = null;
    TreeLinkNode right = null;
    TreeLinkNode next = null;

    TreeLinkNode(int val) {
        this.val = val;
    }
}
