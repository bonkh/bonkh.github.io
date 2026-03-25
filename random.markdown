---
layout: default
title: Random
---

<div class="category-page">
  <h2>Random</h2>
    <p class="category-description">
    Some random thoughts about my daily life
    </p>
  <ul class="post-list">
    {% for post in site.posts %}
      {% if post.category == "random" %}
        <li class="post-item">
          <span class="post-meta">{{ post.date | date: "%b %-d, %Y" }}</span>
          <h3><a href="{{ post.url }}">{{ post.title }}</a></h3>
          <p>{{ post.excerpt }}</p>
        </li>
      {% endif %}
    {% endfor %}
  </ul>
</div>
