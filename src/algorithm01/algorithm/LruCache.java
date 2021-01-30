package algorithm01.algorithm;

import java.util.HashMap;
import java.util.Map;

/**
 * lru cache
 * <p>
 * There are two ways to implement lru caching.One is to use extends LinkedHashMap.
 * This method is relatively simple and only needs to be rewritten.
 * the first implement example such as:
 * <pre>
 * @code
 * public class XX extends LinkedHashMap {
 *     public void set(int key, int val) {
 *         return super.put(key, val);
 *     }
 *
 *     public int get(int key) {
 *         return super.get(key);
 *     }
 * }
 * </pre>
 * The other is to complete lru caching through self-realization doubly linked list, for study,
 * we use the latter method to achieve
 *
 * @author boomzy
 * @date 2021/1/24 16:35
 */
public class LruCache {
    /**
     * basic data structure
     */
    static class LinkNode {
        int key;
        int val;
        LinkNode prev;
        LinkNode next;

        public LinkNode(int key, int val) {
            this.key = key;
            this.val = val;
        }
    }

    /**
     * the capacity of lru cache
     */
    private int capacity;

    /**
     * the container of lru cache
     */
    private Map<Integer, LinkNode> container = new HashMap<>();

    /**
     * the first node of the linked list
     */
    private LinkNode head;
    /**
     * the last node of linked list
     */
    private LinkNode tail;

    /**
     * Constructs an empty <tt>LRUCache</tt> with the specified initial capacity
     *
     * @param capacity the initial capacity of lru cache
     */
    public LruCache(int capacity) {
        this.capacity = capacity;
        head = new LinkNode(0, 0);
        tail = new LinkNode(0, 0);
        head.next = tail;
        tail.prev = head;
    }

    /**
     * Constructs an empty <tt>LRUCache</tt> with the default initial capacity(2)
     */
    public LruCache() {
        this(2);
    }

    /**
     * Returns the value to which the specified key is mapped
     *
     * @param key the specified key
     * @return the value of specified key. <tt>-1</tt> when the key is not mapped
     */
    public int get(int key) {
        if (!container.containsKey(key)) {
            return -1;
        }
        LinkNode node = container.get(key);
        moveNodeToFirst(node);
        return node.val;
    }

    /**
     * Associates the specified value with the specified key in this map
     * and put the entry to container
     * <p>
     * If the map previously contained a mapping for the key, the old
     * value is replaced.
     *
     * @param key the specified key
     * @param val the specified value
     */
    public void set(int key, int val) {
        if (!container.containsKey(key)) {
            if (container.size() == capacity) {
                deleteLastNode();
            }
            LinkNode temp = head.next;
            LinkNode newNode = new LinkNode(key, val);
            head.next = newNode;
            newNode.prev = head;
            newNode.next = temp;
            temp.prev = newNode;
            container.put(key, newNode);
        } else {
            LinkNode node = container.get(key);
            node.val = val;
            moveNodeToFirst(node);
        }
    }

    /**
     * delete the last node of the linked list
     */
    private void deleteLastNode() {
        LinkNode lastNode = tail.prev;
        lastNode.prev.next = tail;
        tail.prev = lastNode.prev;
        container.remove(lastNode.key);
    }

    /**
     * move the specified node to the head of the linked list
     *
     * @param node the node to be moved to the head of the linked list
     */
    private void moveNodeToFirst(LinkNode node) {
        // unbind the node and the left and right nodes
        node.next.prev = node.prev;
        node.prev.next = node.next;
        // move node
        LinkNode temp = head.next;
        head.next = node;
        node.prev = head;
        node.next = temp;
        temp.prev = node;
    }
}