package algorithm01.algorithm;

import java.util.Stack;

/**
 * min stack
 *
 * @author boomzy
 * @date 2021/4/24 10:31
 */
public class MinStack {

    /**
     * 储存元素的栈
     */
    private Stack<Integer> dataStack;

    /**
     * 储存最小数的栈
     */
    private Stack<Integer> minStack;

    public MinStack() {
        dataStack = new Stack<>();
        minStack = new Stack<>();
    }

    /**
     * 入栈
     * <p>
     * 首先判断最小栈是否为空，为空则入栈。如果不为空，则判断要入栈的元素是否比最小栈栈顶元素还小，
     * 是则入栈。最后将元素入数据栈。
     *
     * @param element 入栈元素
     */
    public void push(int element) {
        if (minStack.isEmpty()) {
            minStack.push(element);
        } else if (element <= getMin()) {
            minStack.push(element);
        }
        dataStack.push(element);
    }

    /**
     * 出栈
     * <p>
     * 判断数据栈不为空，满足则取出栈数据栈栈顶元素，如果该元素也是最小栈的栈顶，同时出栈。
     */
    public void pop() {
        if (dataStack.isEmpty()) {
            throw new RuntimeException();
        }
        int pop = dataStack.pop();
        if (pop == getMin()) {
            minStack.pop();
        }
    }

    public int getMin() {
        if (minStack.isEmpty()) {
            throw new RuntimeException();
        }
        return minStack.peek();
    }
}
