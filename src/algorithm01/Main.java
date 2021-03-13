package algorithm01;


import algorithm01.algorithm.LruCache;

import java.util.*;

/**
 * 牛客算法题，频率由高到低
 *
 * @author boomzy
 * @date 2021-01-24 15:30
 */
public class Main {
    /**
     * 78.链表反转
     */
    public ListNode ReverseList(ListNode head) {
        ListNode pre = null;
        ListNode curr = head;
        while (curr != null) {
            ListNode next = curr.next;
            curr.next = pre;
            pre = curr;
            curr = next;
        }
        return pre;
    }

    /**
     * 140.排序
     * <p>
     * 给定一个数组，请你编写一个函数，返回该数组排序后的形式。
     *
     * @param arr
     * @return
     */
    public int[] MySort(int[] arr) {
        // write code here
        return null;
    }

    /**
     * 93.设计LRU缓存结构
     * <p>
     * 设计LRU缓存结构，该结构在构造时确定大小，假设大小为K，并有如下两个功能
     * set(key, value)：将记录(key, value)插入该结构
     * get(key)：返回key对应的value值
     * <p>
     * [要求]
     * set和get方法的时间复杂度为O(1)
     * 某个key的set或get操作一旦发生，认为这个key的记录成了最常使用的。
     * 当缓存的大小超过K时，移除最不经常使用的记录，即set或get最久远的。
     * 若opt=1，接下来两个整数x, y，表示set(x, y)
     * 若opt=2，接下来一个整数x，表示get(x)，若x未出现过或已被移除，则返回-1
     * 对于每个操作2，输出一个答案
     * <p>
     * 示例1
     * <p>
     * 输入
     * [[1,1,1],[1,2,2],[1,3,2],[2,1],[1,4,4],[2,2]],3
     * 返回值
     * [1,-1]
     * 说明
     * 第一次操作后：最常使用的记录为("1", 1)
     * 第二次操作后：最常使用的记录为("2", 2)，("1", 1)变为最不常用的
     * 第三次操作后：最常使用的记录为("3", 2)，("1", 1)还是最不常用的
     * 第四次操作后：最常用的记录为("1", 1)，("2", 2)变为最不常用的
     * 第五次操作后：大小超过了3，所以移除此时最不常使用的记录("2", 2)，加入记录("4", 4)，并且为最常使用的记录，
     * 然后("3", 2)变为最不常使用的记录
     *
     * @param operators
     * @param k
     * @return
     */
    public int[] LRU(int[][] operators, int k) {
        int length = 0;
        for (int i = 0; i < operators.length; i++) {
            if (operators[i][0] == 2) {
                length++;
            }
        }
        int[] result = new int[length];
        int index = 0;
        LruCache cache = new LruCache(k);
        for (int i = 0; i < operators.length; i++) {
            if (operators[i][0] == 1) {
                cache.set(operators[i][1], operators[i][2]);
            }
            if (operators[i][0] == 2) {
                result[index++] = cache.get(operators[i][1]);
            }
        }
        return result;
    }

    /**
     * 4. 判断链表中是否有环
     * 判断给定的链表中是否有环。如果有环则返回true，否则返回false。
     * 你能给出空间复杂度的解法么？
     *
     * @param head
     * @return
     */
    public boolean hasCycle(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        if (fast == null) {
            return false;
        }
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) {
                return true;
            }
        }
        return false;
    }

    // 45题所用
    List<Integer> pre = new ArrayList<>();
    List<Integer> in = new ArrayList<>();
    List<Integer> post = new ArrayList<>();

    /**
     * 45.二叉树先序、中序和后序遍历
     * <p>
     * 分别按照二叉树先序，中序和后序打印所有的节点。
     * 输入{1,2,3}
     * 返回值 [[1,2,3],[2,1,3],[2,3,1]]
     *
     * @param root
     * @return
     */
    public int[][] threeOrders(TreeNode root) {
        List<List<Integer>> ans = new ArrayList<>();
        preOrder(root);
        inOrder(root);
        postOrder(root);
        ans.add(pre);
        ans.add(in);
        ans.add(post);
        int m = ans.size();
        int n = ans.get(0).size();
        int[][] res = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                res[i][j] = ans.get(i).get(j);
            }
        }
        return res;
    }

    /**
     * 先序遍历
     * <p>
     * 算法：设置栈，将根节点入栈，然后循环判断，当栈不为空时，取出栈顶元素加入到结果集中，
     * 然后判断左右子节点是否有元素，有则加入到栈中。由于栈是先进后出，所以先判断右节点
     * 再判断左节点
     *
     * @param root 二叉树
     */
    private void preOrder(TreeNode root) {
        if (root != null) {
            Stack<TreeNode> stack = new Stack<>();
            stack.push(root);
            while (!stack.isEmpty()) {
                root = stack.pop();
                pre.add(root.val);
                if (root.right != null) {
                    stack.push(root.right);
                }
                if (root.left != null) {
                    stack.push(root.left);
                }
            }
        }
    }

    /**
     * 中序遍历
     * <p>
     * 算法：设置一个栈，当栈不为空时且根节点不为空时，将左子节点入栈。当无法入栈时则出栈
     * 并将节点指向右节点。进行下一次操作。
     *
     * @param root 二叉树
     */
    private void inOrder(TreeNode root) {
        if (root != null) {
            Stack<TreeNode> stack = new Stack<>();
            while (!stack.isEmpty() || root != null) {
                if (root != null) {
                    stack.push(root);
                    root = root.left;
                } else {
                    root = stack.pop();
                    in.add(root.val);
                    root = root.right;
                }
            }
        }
    }

    /**
     * 后序遍历
     * <p>
     * 算法：设置两个栈，一个元素栈一个帮助栈。类似前序遍历的算法，但是判断左右子节点时先判断左再判断右。
     * 每次遍历元素栈时，将出栈的元素放到帮助栈中。最终不断出栈帮助栈的元素即是结果。
     *
     * @param root 二叉树
     */
    private void postOrder(TreeNode root) {
        if (root != null) {
            Stack<TreeNode> stack = new Stack<>();
            Stack<TreeNode> help = new Stack<>();
            stack.push(root);
            while (!stack.isEmpty()) {
                root = stack.pop();
                help.push(root);
                if (root.left != null) {
                    stack.push(root.left);
                }
                if (root.right != null) {
                    stack.push(root.right);
                }
            }
            while (!help.isEmpty()) {
                post.add(help.pop().val);
            }
        }
    }

    /**
     * 105.二分查找
     * <p>
     * 请实现有重复数字的升序数组的二分查找
     * 给定一个 元素有序的（升序）整型数组 nums 和一个目标值 target，写一个函数搜索 nums 中的 target，
     * 如果目标值存在返回下标，否则返回 -1
     *
     * @param nums
     * @param target
     * @return
     */
    public int search(int[] nums, int target) {
        int low = 0, high = nums.length - 1;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (nums[mid] == target) {
                while (mid != 0 && (nums[mid - 1] == nums[mid])) {
                    mid--;
                }
                return mid;
            } else if (nums[mid] > target) {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        return -1;
    }

    /**
     * 119.最小的K个数
     * 给定一个数组，找出其中最小的K个数。例如数组元素是4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4。
     * 如果K大于数组的长度，那么返回一个空的数组
     * <p>
     * eg:
     * 输入：[4,5,1,6,2,7,3,8],4
     * 输出：[1,2,3,4]
     *
     * @param input
     * @param k
     * @return
     */
    public ArrayList<Integer> GetLeastNumbers_Solution(int[] input, int k) {
        // 使用快排实现
        ArrayList<Integer> res = new ArrayList<>();
        if (input == null || input.length == 0 || k > input.length) {
            return res;
        }
        quickSort(input, 0, input.length - 1);
        for (int i = 0; i < k; i++) {
            res.add(input[i]);
        }
        return res;
    }

    // 快速排序
    private void quickSort(int[] arr, int start, int end) {
        if (start >= end) {
            return;
        }
        int low = start;
        int high = end;
        int stard = arr[start];
        while (low < high) {
            while (low < high && stard <= arr[high]) {
                high--;
            }
            arr[low] = arr[high];
            while (low < high && stard >= arr[low]) {
                low++;
            }
            arr[high] = arr[low];
        }
        arr[low] = stard;
        quickSort(arr, start, low);
        quickSort(arr, low + 1, end);
    }

    /**
     * 15.二叉树的层次遍历
     * <p>
     * eg:
     * 输入：{1,2}
     * 输出：[[1],[2]]
     *
     * @param root
     * @return
     */
    public ArrayList<ArrayList<Integer>> levelOrder(TreeNode root) {
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            ArrayList<Integer> level = new ArrayList<>();
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode head = queue.poll();
                level.add(head.val);
                if (head.left != null) {
                    queue.offer(head.left);
                }
                if (head.right != null) {
                    queue.offer(head.right);
                }
            }
            res.add(level);
        }
        return res;
    }

    /**
     * 68.跳台阶
     * <p>
     * 一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法
     * （先后次序不同算不同的结果）。
     *
     * @param target
     * @return
     */
    public int jumpFloor(int target) {
        // 动态规划/斐波那契数列问题
        if (target == 0 || target == 1) {
            return target;
        }
        int a = 1, b = 1, c = 0;
        for (int i = 2; i <= target; i++) {
            c = a + b;
            a = b;
            b = c;
        }
        return c;
    }

    /**
     * 19.子数组的最大累加和问题
     * <p>
     * 给定一个数组arr，返回子数组的最大累加和
     * 例如，arr = [1, -2, 3, 5, -2, 6, -1]，所有子数组中，[3, 5, -2, 6]可以累加出最大的和12，所以返回12.
     * 题目保证没有全为负数的数据
     * [要求]
     * 时间复杂度为O(n)，空间复杂度为O(1)
     * <p>
     * eg:
     * 输入：[1, -2, 3, 5, -2, 6, -1]
     * 返回：12
     *
     * @param arr
     * @return
     */
    public int maxsumofSubarray(int[] arr) {
        /*
            动态规划简化而来。
            1.dp定义：dp[i]表示下标为i时的最大累加和
            2.初始化: dp[0] = arr[0]
            3.状态转移方程（i从1开始）：
                if dp[i-1] > 0  ->  dp[i] = dp[i-1] + arr[i]
                if dp[i-1] < 0  ->  dp[i] = arr[i]
            4.结果：res = Math.max(res, dp[i])
         */
        if (arr == null || arr.length == 0) {
            return 0;
        }
        int sum = arr[0];
        int currSum = sum;
        for (int i = 1; i < arr.length; i++) {
            currSum = currSum > 0 ? currSum + arr[i] : arr[i];
            sum = Math.max(currSum, sum);
        }
        return sum;
    }

    /**
     * 61.两数之和
     *
     * @param numbers
     * @param target
     * @return
     */
    public int[] twoSum(int[] numbers, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < numbers.length; i++) {
            int complete = target - numbers[i];
            if (map.containsKey(complete)) {
                return new int[]{map.get(complete), i + 1};
            }
            map.put(numbers[i], i + 1);
        }
        return null;
    }

    /**
     * 33.合并两个有序链表(递归实现)
     *
     * @param l1
     * @param l2
     * @return
     */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) {
            return l2;
        }
        if (l2 == null) {
            return l1;
        }
        ListNode head = null;
        if (l1.val < l2.val) {
            head = l1;
            head.next = mergeTwoLists(l1.next, l2);
        } else {
            head = l2;
            head.next = mergeTwoLists(l1, l2.next);
        }
        return head;
    }

    /**
     * 33.合并两个有序链表(非递归实现)
     *
     * @param l1
     * @param l2
     * @return
     */
    public ListNode mergeTwoListsNonRecursion(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(-1);
        ListNode res = dummy;
        while (l1 != null || l2 != null) {
            if (l1 != null && l2 != null) {
                if (l1.val < l2.val) {
                    dummy.next = l1;
                    l1 = l1.next;
                } else {
                    dummy.next = l2;
                    l2 = l2.next;
                }
                dummy = dummy.next;
            } else if (l1 != null && l2 == null) {
                dummy.next = l1;
                l1 = l1.next;
                dummy = dummy.next;
            } else if (l1 == null && l2 != null) {
                dummy.next = l2;
                l2 = l2.next;
                dummy = dummy.next;
            }
        }
        return res.next;
    }

    /**
     * 76.两个栈实现队列
     */
    static class TwoStackImplementQueue {
        Stack<Integer> stack1 = new Stack<Integer>();
        Stack<Integer> stack2 = new Stack<Integer>();

        public void push(int node) {
            stack1.push(node);
            dump();
        }

        public int pop() {
            if (stack1.isEmpty() && stack2.isEmpty()) {
                throw new RuntimeException();
            }
            dump();
            return stack2.pop();
        }

        private void dump() {
            if (stack2.isEmpty()) {
                while (!stack1.isEmpty()) {
                    stack2.push(stack1.pop());
                }
            }
        }
    }

    /**
     * 50.链表中的节点每k个一组翻转
     * <p>
     * 将给出的链表中的节点每\ k k 个一组翻转，返回翻转后的链表
     * 如果链表中的节点数不是\ k k 的倍数，将最后剩下的节点保持原样
     * 你不能更改节点中的值，只能更改节点本身。
     * 要求空间复杂度O(1)
     * 例如：
     * 给定的链表是1→2→3→4→5
     * 对于 k=2, 你应该返回2→1→4→3→5
     * 对于 k=3, 你应该返回3→2→1→4→5
     *
     * @param head
     * @param k
     * @return
     */
    public ListNode reverseKGroup(ListNode head, int k) {
        // 方法一，使用栈
        /*if (head == null) {
            return null;
        }
        Stack<ListNode> stack = new Stack<>();
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode curr = dummy;
        ListNode next = dummy.next;
        while (next != null) {
            for (int i = 0; i < k && next != null; i++) {
                stack.push(next);
                next = next.next;
            }
            if (stack.size() != k) {
                return dummy.next;
            }
            while (!stack.isEmpty()) {
                curr.next = stack.pop();
                curr = curr.next;
            }
            curr.next = next;
        }
        return dummy.next;*/

        // 方法二：两个指针，一个prev指向dummy，一个last在后面，要翻转的是prev和last之间的链表
        if (head == null) {
            return null;
        }
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode prev = dummy;
        while (prev != null) {
            prev = reverse(prev, k);
        }
        return dummy.next;
    }

    private ListNode reverse(ListNode prev, int k) {
        // 先让last往后走k个
        ListNode last = prev;
        for (int i = 0; i < k + 1; i++) {
            last = last.next;
            if (i != k && last == null) {
                return null;
            }
        }

        ListNode tail = prev.next;
        ListNode curr = prev.next.next;
        while (curr != last) {
            ListNode next = curr.next;
            curr.next = prev.next;
            prev.next = curr;
            tail.next = next;
            curr = next;
        }
        return tail;
    }

    /**
     * 41.最长无重复子串
     * <p>
     * 给定一个数组arr，返回arr的最长无的重复子串的长度(无重复指的是所有数字都不相同)。
     * eg:
     * 输入[2,3,4,5]，输出4
     * 输入[2,2,3,4,3]，输出3
     *
     * @param arr
     * @return
     */
    public int maxLength(int[] arr) {
        /*
            方法一：HashSet + 双指针
            使用双指针，快指针j，慢指针i。让j走，如果集合里不包含
            当前字符，则加到字符里。如果包含了，则走慢指针i，不断的判断
            i指针对应的字符是否出现在集合里，出现则移除掉，同时指针
            右移。最后看set的size。即res = max{res, set.size()}
         */
        /*if (arr == null || arr.length == 0) {
            return 0;
        }
        int res = 0;
        HashSet<Integer> set = new HashSet<>();
        for (int i = 0, j = 0; j < arr.length; j++) {
            while (set.contains(arr[j])) {
                set.remove(arr[i]);
                i++;
            }
            set.add(arr[j]);
            res = Math.max(res, set.size());
        }
        return res;*/

        /*
            方法二：HashMap+双指针(优化)。
            方法一需要让慢指针每次走一格，比较次数过多。
            使用双指针，快指针j，慢指针i。让j走，如果集合里不包含
            当前字符，则加到字符里。如果包含了，直接让慢指针走到下
            标为重复字符的下标加1。但是如果重复在慢指针指的之前出现
            了，则判断一下有没有必要回去了。即：
            i被移动到的新位置为max{i, 重复字符出现的位置+1}
         */
        if (arr == null || arr.length == 0) {
            return 0;
        }
        int res = 0;
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0, j = 0; j < arr.length; j++) {
            if (map.containsKey(arr[j])) {
                // 移动到新位置，并且判断是否需要往回走
                i = Math.max(i, map.get(arr[j]) + 1);
            }
            map.put(arr[j], j);
            res = Math.max(res, j - i + 1);
        }
        return res;
    }

}

class ListNode {
    int val;
    ListNode next;

    ListNode(int val) {
        this.val = val;
    }
}


class TreeNode {
    int val = 0;
    TreeNode left = null;
    TreeNode right = null;

    TreeNode(int val) {
        this.val = val;
    }
}

