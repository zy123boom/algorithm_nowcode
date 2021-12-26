package jian_zhi_offer;

import java.util.Arrays;
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
