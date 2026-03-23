---
layout: default
title: Data Science
---

<div class="category-page">
  <h2>Data Science Projects</h2>
  <ul class="post-list">
    {% for post in site.posts %}
      {% if post.category == "data-science" %}
        <li class="post-item">
          <span class="post-meta">{{ post.date | date: "%b %-d, %Y" }}</span>
          <h3><a href="{{ post.url }}">{{ post.title }}</a></h3>
          <p>{{ post.excerpt }}</p>
        </li>
      {% endif %}
    {% endfor %}
  </ul>
</div>
