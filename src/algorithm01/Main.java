package algorithm01;


import algorithm01.algorithm.LruCache;
import algorithm01.algorithm.MinStack;

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
            } else if (l1 != null) {
                dummy.next = l1;
                l1 = l1.next;
                dummy = dummy.next;
            } else {
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

    /**
     * 22.合并两个有序的数组
     * <p>
     * 给出两个有序的整数数组A和B，请将数组B合并到数组A中，变成一个有序的数组
     * 注意：
     * 可以假设A数组有足够的空间存放B数组的元素，A和B中初始的元素数目分别为m和n
     *
     * @param A
     * @param m
     * @param B
     * @param n
     */
    public void merge(int A[], int m, int B[], int n) {
        /*
            算法：由于题目中说A数组有足够的空间存放B数组的元素，所以不必开辟额外空间，直接操作A数组
            使用双指针。
            1.使用指针i指向A的m-1，指针j指向B的n-1。两个指针移动前需要定义一个index=m+n-1代表合并
            数组的最后一个位置
            2.然后进行指针移动，A[i]和B[j]哪个大，就合并哪个。即将该元素放到index位置上。然后index--
            3.最后需要判断B是否合并完成，如果没有合并完，则将B直接放进A
         */
        int i = m - 1, j = n - 1;
        int index = m + n - 1;
        // 步骤2
        while (i >= 0 && j >= 0) {
            A[index--] = A[i] > B[j] ? A[i--] : B[j--];
        }
        // 步骤3
        while (j >= 0) {
            A[index--] = B[j--];
        }
    }

    /**
     * 3.链表中环的入口节点
     * <p>
     * 对于一个给定的链表，返回环的入口节点，如果没有环，返回null
     * 拓展：
     * 你能给出不利用额外空间的解法么？
     *
     * @param head
     * @return
     */
    public ListNode detectCycle(ListNode head) {
        /*
            算法一：使用set，入口节点即为会重复的节点
         */
        /*HashSet<ListNode> set = new HashSet<>();
        while (head != null) {
            if (set.contains(head)) {
                return head;
            }
            set.add(head);
            head = head.next;
        }
        return null;*/

        /*
            算法二：快慢指针判环，如果有环，则让fast指针回到开头，变为每次走一步，
            然后两个指针一起走，相遇处就是入环节点
         */
        if (head == null || head.next == null || head.next.next == null) {
            return null;
        }
        ListNode slow = head.next;
        ListNode fast = head.next.next;
        while (slow != fast) {
            if (fast.next == null || fast.next.next == null) {
                return null;
            }
            slow = slow.next;
            fast = fast.next.next;
        }
        // 有环，让fast指针回到开头然后两个指针一起走
        fast = head;
        while (slow != fast) {
            slow = slow.next;
            fast = fast.next;
        }
        return slow;
    }

    /**
     * 52.括号序列
     * <p>
     * 给出一个仅包含字符'(',')','{','}','['和']',的字符串，判断给出的字符串是否是合法的括号序列
     * 括号必须以正确的顺序关闭，"()"和"()[]{}"都是合法的括号序列，但"(]"和"([)]"不合法。
     *
     * @param s
     * @return
     */
    public boolean isValid(String s) {
        if (s == null || "".equals(s)) {
            return false;
        }
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(' || s.charAt(i) == '[' || s.charAt(i) == '{') {
                stack.push(s.charAt(i));
            } else if (s.charAt(i) == ')' || s.charAt(i) == ']' || s.charAt(i) == '}') {
                if (stack.isEmpty()) {
                    return false;
                }
                char curr = stack.pop();
                if (curr == '(' && s.charAt(i) != ')') {
                    return false;
                }
                if (curr == '[' && s.charAt(i) != ']') {
                    return false;
                }
                if (curr == '{' && s.charAt(i) != '}') {
                    return false;
                }
            }
        }
        return stack.isEmpty();
    }

    /**
     * 1.大数加法
     * <p>
     * 以字符串的形式读入两个数字，编写一个函数计算它们的和，以字符串形式返回。
     * （字符串长度不大于100000，保证字符串仅由'0'~'9'这10种字符组成）
     * <p>
     * eg:
     * 输入 "1","99"
     * 输出 "100"
     *
     * @param s
     * @param t
     * @return
     */
    public String solve(String s, String t) {
        /*
            算法：
            1.首先保证s是长的字符串，如果不是交换两个字符串
            2.new一个StringBuilder保存结果。获取两个字符串长度，记作shortLen, longLen。设置进位carry
            3.做循环的逐位计算和，条件是从0~shortLen。然后获取每一位的值的和加上carry，然后变换carry为下一次循环准备
            4.做循环，条件从短长长度~longLen，取长串s的当前位和carry,变换carry为下一次循环准备
            5.最后判断carry是否为0，不为0则加入到结果中。最终反转StringBuilder即可。
            注明：获取每一个字符的数字值是  cr - '0'。每一位是length - 1 - i
         */
        if (s == null || "".equals(s)) {
            return t;
        }
        if (t == null || "".equals(t)) {
            return s;
        }
        StringBuilder ans = new StringBuilder();
        if (s.length() < t.length()) {
            String temp = s;
            s = t;
            t = temp;
        }
        int longLen = s.length();
        int shortLen = t.length();
        int carry = 0;
        for (int i = 0; i < shortLen; i++) {
            int sum = (s.charAt(longLen - 1 - i) - '0') + (t.charAt(shortLen - 1 - i) - '0') + carry;
            ans.append(sum % 10);
            carry = sum / 10;
        }
        for (int i = shortLen; i < longLen; i++) {
            int sum = s.charAt(longLen - 1 - i) - '0' + carry;
            ans.append(sum % 10);
            carry = sum / 10;
        }
        if (carry != 0) {
            ans.append(carry);
        }
        return ans.reverse().toString();
    }

    /**
     * 53.删除链表的倒数第n个节点
     * <p>
     * 给定一个链表，删除链表的倒数第n个节点并返回链表的头指针
     * 例如：
     * 给出的链表为:1->2->3->4->5, n= 2.
     * 删除了链表的倒数第n个节点之后,链表变为1->2->3->5.
     * 备注：
     * 题目保证n一定是有效的
     * 请给出请给出时间复杂度为O(n)的算法
     *
     * @param head
     * @param n
     * @return
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        /*
            算法：快慢指针法。
            首先让快指针移动n个位置，然后同时移动快慢指针，此时慢指针指向要
            移除的前一个元素，然后删除这个元素，即node.next = node.next.next
         */
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode fast = dummy;
        ListNode slow = dummy;
        for (int i = 0; i < n; i++) {
            fast = fast.next;
        }
        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }
        slow.next = slow.next.next;
        return dummy.next;
    }

    /**
     * 127.最长公共子串
     * <p>
     * 给定两个字符串str1和str2,输出两个字符串的最长公共子串
     * 题目保证str1和str2的最长公共子串存在且唯一。
     * eg:
     * 输入"1AB2345CD","12345EF"
     * 输出"2345"
     *
     * @param str1
     * @param str2
     * @return
     */
    public String LCS(String str1, String str2) {
        // write code here
        // TODO
        return null;
    }

    /**
     * 14.二叉树的之字遍历
     * 给定一个二叉树，返回该二叉树的之字形层序遍历，（第一层从左向右，下一层从右向左，一直这样交替）
     * <p>
     * 例如：
     * 给定的二叉树是{3,9,20,#,#,15,7},
     * 该二叉树之字形层序遍历的结果是
     * [
     * [3],
     * [20,9],
     * [15,7]
     * ]
     *
     * @param root
     * @return
     */
    public ArrayList<ArrayList<Integer>> zigzagLevelOrder(TreeNode root) {
        /*
            使用两个栈实现
         */
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        // 层数
        int layer = 1;
        // 奇数层，正向
        Stack<TreeNode> stack1 = new Stack<>();
        // 偶数层，逆向
        Stack<TreeNode> stack2 = new Stack<>();
        stack1.push(root);
        while (!stack1.isEmpty() || !stack2.isEmpty()) {
            // 奇数
            if (layer % 2 != 0) {
                ArrayList<Integer> curr = new ArrayList<>();
                while (!stack1.isEmpty()) {
                    TreeNode node = stack1.pop();
                    if (node != null) {
                        curr.add(node.val);
                        if (node.left != null) {
                            stack2.push(node.left);
                        }
                        if (node.right != null) {
                            stack2.push(node.right);
                        }
                    }
                }
                if (!curr.isEmpty()) {
                    res.add(curr);
                    layer++;
                }
            } else {
                // 偶数
                ArrayList<Integer> curr = new ArrayList<>();
                while (!stack2.isEmpty()) {
                    TreeNode node = stack2.pop();
                    if (node != null) {
                        curr.add(node.val);
                        if (node.right != null) {
                            stack1.push(node.right);
                        }
                        if (node.left != null) {
                            stack1.push(node.left);
                        }
                    }
                }
                if (!curr.isEmpty()) {
                    res.add(curr);
                    layer++;
                }
            }
        }
        return res;
    }

    /**
     * 40.两个链表生成相加链表
     * <p>
     * 假设链表中每一个节点的值都在 0 - 9 之间，那么链表整体就可以代表一个整数。
     * 给定两个这种链表，请生成代表两个整数相加值的结果链表。
     * 例如：链表 1 为 9->3->7，链表 2 为 6->3，最后生成新的结果链表为 1->0->0->0。
     *
     * @param head1
     * @param head2
     * @return
     */
    public ListNode addInList(ListNode head1, ListNode head2) {
        /*
            三个阶段。第一是处理两个链表都有的公共部分。例如例子中 3 -> 7和6 -> 3
            第二是处理一个有一个没有的，例如例子中的链表1的 9
            第三是处理最后可能出现进位的carry。
            其中，对于第一阶段：
            当前位sum = l1 + l2 + carry
            当前位val = sum % 10
            carry = sum / 10

            对于第二阶段：
            当前位sum = l1 + carry
            当前位val = sum % 10
            carry = sum / 10

            对于第三阶段：
            如果carry = 1, new一个新节点，值为1
         */
        ListNode dummy = new ListNode(0);
        int sum = 0;
        ListNode cur = dummy;
        ListNode p1 = head1, p2 = head2;
        while (p1 != null || p2 != null) {
            if (p1 != null) {
                sum += p1.val;
                p1 = p1.next;
            }
            if (p2 != null) {
                sum += p2.val;
                p2 = p2.next;
            }
            cur.next = new ListNode(sum % 10);
            sum /= 10;
            cur = cur.next;
        }

        if (sum == 1) {
            cur.next = new ListNode(1);
        }

        return dummy.next;
    }

    /**
     * 102.最近公共祖先
     * <p>
     * 给定一棵二叉树以及这棵树上的两个节点 o1 和 o2，请找到 o1 和 o2 的最近公共祖先节点。
     * 输入：
     * [3,5,1,6,2,0,8,#,#,7,4],5,1
     * 输出：
     * 3
     *
     * @param root
     * @param o1
     * @param o2
     * @return
     */
    public int lowestCommonAncestor(TreeNode root, int o1, int o2) {
        /*
            最近公共祖先和o1,o2有三种关系：

            o1,o2分别在祖先左右两侧
            祖先是o1，o2在祖先左/右侧
            祖先是o2，o1在祖先左/右侧
            使用dfs深度遍历，如果节点为o1,o2中其中一个直接返回，如果节点超过叶子节点也返回
         */
        return commonAncestor(root, o1, o2).val;
    }

    private TreeNode commonAncestor(TreeNode root, int o1, int o2) {
        // 如果节点为o1,o2中其中一个直接返回，节点超过叶子节点也返回
        if (root == null || root.val == o1 || root.val == o2) {
            return root;
        }
        TreeNode left = commonAncestor(root.left, o1, o2);
        TreeNode right = commonAncestor(root.right, o1, o2);
        if (left == null) {
            return right;
        }
        if (right == null) {
            return left;
        }
        return root;
    }

    /**
     * 38.螺旋矩阵
     * <p>
     * 给定一个m x n大小的矩阵（m行，n列），按螺旋的顺序返回矩阵中的所有元素。
     * eg:
     * 输入[[1,2,3],[4,5,6],[7,8,9]]
     * 输出[1,2,3,6,9,8,7,4,5]
     *
     * @param matrix
     * @return
     */
    public ArrayList<Integer> spiralOrder(int[][] matrix) {
        ArrayList<Integer> res = new ArrayList<>();
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return res;
        }
        int top = 0, bottom = matrix.length - 1;
        int left = 0, right = matrix[0].length - 1;
        while (top < bottom && left < right) {
            for (int i = left; i < right; i++) {
                res.add(matrix[top][i]);
            }
            for (int i = top; i < bottom; i++) {
                res.add(matrix[i][right]);
            }
            for (int i = right; i > left; i--) {
                res.add(matrix[bottom][i]);
            }
            for (int i = bottom; i > top; i--) {
                res.add(matrix[i][left]);
            }
            top++;
            right--;
            bottom--;
            left++;
        }

        // 特殊情况：只剩一行或者一列（含只剩一个的情况）
        if (top == bottom) {
            for (int i = left; i <= right; i++) {
                res.add(matrix[top][i]);
            }
            return res;
        }
        if (left == right) {
            for (int i = top; i <= bottom; i++) {
                res.add(matrix[i][left]);
            }
            return res;
        }

        return res;
    }

    /**
     * 65.斐波那契数列
     *
     * @param n
     * @return
     */
    public int Fibonacci(int n) {
        if (n <= 1) {
            return n;
        }
        int a = 0, b = 1;
        int c = 0;
        for (int i = 2; i <= n; i++) {
            c = a + b;
            a = b;
            b = c;
        }
        return c;
    }

    /**
     * 17.最长回文子串
     * <p>
     * 对于一个字符串，请设计一个高效算法，计算其中最长回文子串的长度。
     * 给定字符串A以及它的长度n，请返回最长回文子串的长度。
     * eg:
     * 输入："abc1234321ab",12
     * 输出：7
     *
     * @param A
     * @param n
     * @return
     */
    public int getLongestPalindrome(String A, int n) {
        /*
            中心扩散。分为A为奇数个数和偶数个数两种。如果是奇数，左是i-1,右是i+1.
            如果是偶数，左是i,右是i+1
         */
        if (A == null || n == 0) {
            return 0;
        }
        int max = 0;
        // 最终子串的左右边界
        int ll = 0, rr = 0;
        for (int i = 0; i < n; i++) {
            // 奇数情况
            int l = i - 1;
            int r = i + 1;
            while (l >= 0 && r < n && A.charAt(l) == A.charAt(r)) {
                int len = r - l + 1;
                if (len > max) {
                    max = len;
                    ll = l;
                    rr = r;
                }
                l--;
                r++;
            }
            // 偶数情况，即奇数情况的while不满足
            l = i;
            r = i + 1;
            while (l >= 0 && r < n && A.charAt(l) == A.charAt(r)) {
                int len = r - l + 1;
                if (len > max) {
                    max = len;
                    ll = l;
                    rr = r;
                }
                l--;
                r++;
            }
        }
        return A.substring(ll, rr + 1).length();
    }

    /**
     * 54.数组中相加和为0的三元组
     * <p>
     * 给出一个有n个元素的数组S，S中是否有元素a,b,c满足a+b+c=0？找出数组S中所有满足条件的三元组。
     * 注意：
     * 三元组（a、b、c）中的元素必须按非降序排列。（即a≤b≤c）
     * 解集中不能包含重复的三元组。
     * 例如，给定的数组 S = {-10 0 10 20 -10 -40},解集为(-10, 0, 10) (-10, -10, 20)
     *
     * @param num
     * @return
     */
    public ArrayList<ArrayList<Integer>> threeSum(int[] num) {
        /*
            思路：
            （1）首先对数组进行排序（从小到大）
            （2）依次取出第 i 个数（i从0开始），并且不重复的选取（跳过重复的数）
            （3）这样问题就转换为 2 个数求和的问题（可以用双指针解决方法）
            ==》数求和问题
            （4）定义两个指针：左指针（left） 和 右指针（right）
            （5）找出固定 left， 此时left所指的位置为数组中最小数，再找到两个数和 不大于 target 的最大 right 的位置
            （6）调整 left 的位置（后移），求解和是否为 target O(n)
            ==》时间复杂度：O(nlogn) + O(n)
         */
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        if (num == null || num.length < 3) {
            return res;
        }
        Arrays.sort(num);
        for (int i = 0; i < num.length - 2; i++) {
            // 如果当前数字大于0，且数组已经排过序，则三数之和一定大于0，所以结束循环
            if (num[i] > 0) {
                break;
            }
            // 去重
            if (i > 0 && num[i] == num[i - 1]) {
                continue;
            }
            int left = i + 1;
            int right = num.length - 1;
            while (left < right) {
                int sum = num[i] + num[left] + num[right];
                if (sum == 0) {
                    ArrayList<Integer> curr = new ArrayList<>();
                    curr.add(num[i]);
                    curr.add(num[left]);
                    curr.add(num[right]);
                    res.add(curr);
                    while (left < right && num[left] == num[left + 1]) {
                        left++;
                    }
                    while (left < right && num[right] == num[right - 1]) {
                        right--;
                    }
                    left++;
                    right--;
                } else if (sum > 0) {
                    right--;
                } else {
                    left++;
                }
            }
        }
        return res;
    }

    /**
     * 12.重建二叉树
     * <p>
     * 输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序
     * 遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}
     * 和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
     *
     * @param pre
     * @param in
     * @return
     */
    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        /*
            第一步：获取根节点
            第二步：构建left左子树和right右子树，递归构建。
            root.left = reConstructBinaryTree(左子树的前序数组，左子树的中序数组)
            root.right = reConstructBinaryTree(右子树的前序数组，右子树的中序数组)
                其中，构建的时候先找到根节点在中序数组中的位置，这样就可知道左子树和右子树在数组中是什么，
                然后就可以根据这个信息在前序遍历的数组中找到对应的左子树和右子树。
         */
        if (pre == null || pre.length == 0) {
            return null;
        }

        // 1.获取根节点
        TreeNode root = new TreeNode(pre[0]);
        // 2.1 获取根节点在中序数组中的位置
        int index = findIndex(pre, in);

        // 2.2构建左子树和右子树
        root.left = reConstructBinaryTree(Arrays.copyOfRange(pre, 1, index + 1),
                Arrays.copyOfRange(in, 0, index));
        root.right = reConstructBinaryTree(Arrays.copyOfRange(pre, index + 1, pre.length),
                Arrays.copyOfRange(in, index + 1, in.length));
        // 3.返回
        return root;
    }

    private int findIndex(int[] pre, int[] in) {
        for (int i = 0; i < in.length; i++) {
            if (in[i] == pre[0]) {
                return i;
            }
        }
        return 0;
    }

    /**
     * 32.求平方根
     * <p>
     * 实现函数 int sqrt(int x).
     * 计算并返回x的平方根（向下取整）
     *
     * @param x
     * @return
     */
    public int sqrt(int x) {
        // 二分
        if (x <= 0) {
            return 0;
        }
        int magicNum = (int) Math.sqrt(Integer.MAX_VALUE);
        int start = 0, end = magicNum;
        while (start + 1 < end) {
            int mid = start + (end - start) / 2;
            if (mid * mid == x) {
                return mid;
            }
            if (mid * mid > x) {
                end = mid;
            } else {
                start = mid;
            }
        }

        if (end * end == x) {
            return end;
        } else {
            return start;
        }
    }

    /**
     * 90.设计getMin功能的栈
     * <p>
     * 实现一个特殊功能的栈，在实现栈的基本功能的基础上，再实现返回栈中最小元素的操作。
     * eg:
     * 输入[[1,3],[1,2],[1,1],[3],[2],[3]]
     * 输出[1,2]
     * <p>
     * 有三种操作种类，op1表示push，op2表示pop，op3表示getMin。你需要返回和op3出现次数一样多的数组，
     * 表示每次getMin的答案
     * <p>
     * 1<=操作总数<=1000000
     * -1000000<=每个操作数<=1000000
     * 数据保证没有不合法的操作
     *
     * @param op
     * @return
     */
    public int[] getMinStack(int[][] op) {
        int len = 0;
        for (int i = 0; i < op.length; i++) {
            if (op[i][0] == 3) {
                len++;
            }
        }
        int[] res = new int[len];
        MinStack minStack = new MinStack();
        int index = 0;
        for (int i = 0; i < op.length; i++) {
            if (op[i][0] == 1) {
                minStack.push(op[i][1]);
                continue;
            }
            if (op[i][0] == 2) {
                minStack.pop();
                continue;
            }
            if (op[i][0] == 3) {
                res[index++] = minStack.getMin();
            }
        }
        return res;
    }

    /**
     * 128.容器盛水问题
     * <p>
     * 给定一个整形数组arr，已知其中所有的值都是非负的，将这个数组看作一个容器，
     * 请返回容器能装多少水。
     * eg:
     * 输入[3,1,2,5,2,4]
     * 输出5
     *
     * @param arr
     * @return
     */
    public long maxWater(int[] arr) {
        /*
            双指针算法，一个指向左边界一个指向右边界。
            找到一个最小的边界，如果左边界最小则从左向右移动，并且计算容量。
            如果右边界最小则从右向左移动，并且计算容量。容量为高度差。
         */
        if (arr == null || arr.length < 2) {
            return 0;
        }
        long res = 0L;
        int left = 0, right = arr.length - 1;
        while (left < right) {
            int currMin = Math.min(arr[left], arr[right]);
            while (left < right && arr[left] <= currMin) {
                res += currMin - arr[left];
                left++;
            }
            while (left < right && arr[right] <= currMin) {
                res += currMin - arr[right];
                right--;
            }
        }
        return res;
    }

    /**
     * 136.输出二叉树的右视图
     * <p>
     * 请根据二叉树的前序遍历，中序遍历恢复二叉树，并打印出二叉树的右视图
     * 输入：[1,2,4,5,3],[4,2,5,1,3]
     * 输出：[1,3,5]
     *
     * @param xianxu
     * @param zhongxu
     * @return
     */
    public int[] solve(int[] xianxu, int[] zhongxu) {
        /*
            1.构建二叉树
            2.层次遍历
         */
        TreeNode head = reConstructBinaryTree2(xianxu, zhongxu);
        return buildRightSideView(head);
    }

    /**
     * 136帮助函数 构建二叉树
     *
     * @param xianxu
     * @param zhongxu
     * @return
     */
    private TreeNode reConstructBinaryTree2(int[] xianxu, int[] zhongxu) {
//        reConstructBinaryTree(xianxu, zhongxu);
        /*
            第一步：获取根节点
            第二步：构建left左子树和right右子树，递归构建。
            root.left = reConstructBinaryTree(左子树的前序数组，左子树的中序数组)
            root.right = reConstructBinaryTree(右子树的前序数组，右子树的中序数组)
                其中，构建的时候先找到根节点在中序数组中的位置，这样就可知道左子树和右子树在数组中是什么，
                然后就可以根据这个信息在前序遍历的数组中找到对应的左子树和右子树。
         */
        if (xianxu == null || xianxu.length == 0) {
            return null;
        }
        TreeNode root = new TreeNode(xianxu[0]);
        int index = findIndex2(xianxu, zhongxu);

        root.left = reConstructBinaryTree2(Arrays.copyOfRange(xianxu, 1, index + 1),
                Arrays.copyOfRange(zhongxu, 0, index));
        root.right = reConstructBinaryTree2(Arrays.copyOfRange(xianxu, index + 1, xianxu.length),
                Arrays.copyOfRange(zhongxu, index + 1, zhongxu.length));
        return root;
    }

    private int findIndex2(int[] xianxu, int[] zhongxu) {
        for (int i = 0; i < zhongxu.length; i++) {
            if (zhongxu[i] == xianxu[0]) {
                return i;
            }
        }
        return 0;
    }

    /**
     * 136帮助函数，构建右视图
     *
     * @param head
     * @return
     */
    private int[] buildRightSideView(TreeNode head) {
        List<Integer> res = new ArrayList<>();
        if (head == null) {
            return new int[0];
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(head);
        while (!queue.isEmpty()) {
            TreeNode temp = head;
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                temp = queue.poll();
                if (temp.left != null) {
                    queue.offer(temp.left);
                }
                if (temp.right != null) {
                    queue.offer(temp.right);
                }
            }
            // 添加每层最右元素
            res.add(temp.val);
        }
        return res.stream().mapToInt(Integer::valueOf).toArray();
    }

    /**
     * 7.买卖股票的最好时机
     * <p>
     * 假设你有一个数组，其中第i个元素是股票在第i天的价格。
     * 你有一次买入和卖出的机会。（只有买入了股票以后才能卖出）。请你设
     * 计一个算法来计算可以获得的最大收益。
     *
     * @param prices
     * @return
     */
    public int maxProfit(int[] prices) {
        int minPrice = Integer.MAX_VALUE;
        int maxProfit = 0;
        for (int i = 0; i < prices.length; i++) {
            if (prices[i] < minPrice) {
                minPrice = prices[i];
            } else if (prices[i] - minPrice > maxProfit) {
                maxProfit = prices[i] - minPrice;
            }
        }
        return maxProfit;
    }

    /**
     * 51.合并K个排序链表(优先队列解法)
     *
     * @param lists
     * @return
     */
    public ListNode mergeKListsByPriorityQueue(ArrayList<ListNode> lists) {
        /*
            使用优先队列，设置dummy。首先将lists中所有链表的第一个节点放入优先队列。
            然后当队列不为空时出队，此时出队元素就是最小的。然后指针指向下一个，然后判断出队的node.next是否
            为空，不为空则入队node.next
         */
        ListNode dummy = new ListNode(0);
        if (lists == null || lists.size() == 0) {
            return dummy.next;
        }
        ListNode current = dummy;
        PriorityQueue<ListNode> priorityQueue = new PriorityQueue<>((o1, o2) -> o1.val - o2.val);
        for (ListNode node : lists) {
            if (node != null) {
                priorityQueue.add(node);
            }
        }

        while (!priorityQueue.isEmpty()) {
            ListNode node = priorityQueue.poll();
            current.next = node;
            current = current.next;
            if (node.next != null) {
                priorityQueue.add(node.next);
            }
        }

        return dummy.next;
    }

    /**
     * 51.合并K个排序链表
     * <p>
     * 合并k个已排序的链表并将其作为一个已排序的链表返回。分析并描述其复杂度。
     * eg:
     * 输入[{1,2,3},{4,5,6,7}]
     * 输出{1,2,3,4,5,6,7}
     *
     * @param lists
     * @return
     */
    public ListNode mergeKLists(ArrayList<ListNode> lists) {
        /*
            分治思想，先分开，到不能分的时候，然后做合并两个有序链表的操作。
         */
        return merge(lists, 0, lists.size() - 1);
    }

    private ListNode merge(ArrayList<ListNode> lists, int left, int right) {
        // base case
        if (left == right) {
            return lists.get(left);
        }
        if (left > right) {
            return null;
        }

        int mid = left + (right - left) / 2;
        return mergeTwoLists(merge(lists, left, mid), merge(lists, mid + 1, right));
    }

    /**
     * 121.字符串的排列
     * <p>
     * 题目描述
     * 输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则按字典序打印出
     * 由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。
     * <p>
     * eg:
     * 输入"ab"
     * 输出["ab","ba"]
     *
     * @param str
     * @return
     */
    public ArrayList<String> Permutation(String str) {
        // 全排列
        if (str == null || "".equals(str)) {
            return new ArrayList<>();
        }
        char[] charArr = str.toCharArray();
        Set<String> set = new HashSet<>();
        helper(set, new StringBuilder(), charArr, new boolean[charArr.length]);
        ArrayList<String> res = new ArrayList<>(set);
        Collections.sort(res);
        return res;
    }

    private void helper(Set<String> res, StringBuilder sb, char[] charArr, boolean[] visited) {
        if (sb.length() == charArr.length) {
            res.add(sb.toString());
            return;
        }
        for (int i = 0; i < charArr.length; i++) {
            if (visited[i]) {
                continue;
            }
            visited[i] = true;
            sb.append(charArr[i]);
            helper(res, sb, charArr, visited);
            sb.deleteCharAt(sb.length() - 1);
            visited[i] = false;
        }
    }

    // 109题
    private int[][] direction = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    /**
     * 109.判断岛屿数量
     * <p>
     * 给一个01矩阵，1代表是陆地，0代表海洋， 如果两个1相邻，那么这两个1属于同一个岛。我们只考虑上下左右为相邻。
     * 岛屿: 相邻陆地可以组成一个岛屿（相邻:上下左右） 判断岛屿个数。
     * <p>
     * eg:
     * 输入
     * [
     * [1,1,0,0,0],
     * [0,1,0,1,1],
     * [0,0,0,1,1],
     * [0,0,0,0,0],
     * [0,0,1,1,1]
     * ]
     * 输出3
     *
     * @param grid char字符型二维数组
     * @return int整型
     */
    public int solve(char[][] grid) {
        // dfs
        if (grid == null || grid.length == 0 || grid[0].length == 0) {
            return 0;
        }
        int res = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] != '0') {
                    dfs(grid, i, j);
                    res++;
                }
            }
        }
        return res;
    }

    private void dfs(char[][] grid, int row, int column) {
        if (row < 0 || row >= grid.length || column < 0 || column >= grid[0].length || grid[row][column] == '0') {
            return;
        }

        grid[row][column] = '0';
        for (int[] d : direction) {
            dfs(grid, row + d[0], column + d[1]);
        }
    }

    /**
     * 13.二叉树的最大深度
     * <p>
     * 求给定二叉树的最大深度，
     * 最大深度是指树的根结点到最远叶子结点的最长路径上结点的数量。
     *
     * @param root
     * @return
     */
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int leftDepth = maxDepth(root.left);
        int rightDepth = maxDepth(root.right);
        return Math.max(leftDepth, rightDepth) + 1;
    }

    /**
     * 70.链表排序
     * <p>
     * 给定一个无序单链表，实现单链表的排序(按升序排序)。
     *
     * @param head
     * @return
     */
    public ListNode sortInList(ListNode head) {
        /*
            1.快慢指针法找中点
            2.归并排序
         */
        return mergeSort(head);
    }

    private ListNode mergeSort(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        // 1.
        ListNode fast = dummy;
        ListNode slow = dummy;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }

        ListNode head2 = slow.next;
        slow.next = null;
        head = mergeSort(head);
        head2 = mergeSort(head2);
        return merge(head, head2);
    }

    private ListNode merge(ListNode head1, ListNode head2) {
        ListNode dummy = new ListNode(0);
        ListNode curr = dummy;
        while (head1 != null && head2 != null) {
            if (head1.val < head2.val) {
                curr.next = head1;
                curr = curr.next;
                head1 = head1.next;
            } else {
                curr.next = head2;
                curr = curr.next;
                head2 = head2.next;
            }
        }
        if (head1 != null) {
            curr.next = head1;
        }
        if (head2 != null) {
            curr.next = head2;
        }
        return dummy.next;
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

