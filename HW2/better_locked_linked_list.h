#ifndef _BETTER_LOCKED_LINKED_LIST_H_
#define _BETTER_LOCKED_LINKED_LIST_H_

#include <iostream>
#include <mutex>
#include <thread>
#include "linked_list.h"

// basic node structure
class better_locked_linked_list : public linked_list {
  class node {
    // TODO
  };
  node *head;
 public:
  better_locked_linked_list() {
    // TODO
  }

  bool contains(int key) {
    // TODO
  }

  bool insert(int key) {
    // TODO
  }

  bool remove(int key) {
    // TODO
  }
};
#endif
