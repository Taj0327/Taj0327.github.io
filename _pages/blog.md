---
layout: default
permalink: /blog/
title: blog
nav: true
nav_order: 1
---

<div class="post">

  {% assign postlist = site.posts %}

  {% if postlist and postlist.size > 0 %}
    <ul class="post-list">
      {% for post in postlist %}
        <li>
          <h3>
            <a class="post-title" href="{{ post.url | relative_url }}">{{ post.title }}</a>
          </h3>
          {% if post.description %}
            <p>{{ post.description }}</p>
          {% endif %}
          <p class="post-meta">
            {{ post.date | date: '%B %d, %Y' }}
          </p>
        </li>
      {% endfor %}
    </ul>
  {% else %}
    <p>No posts yet. Check back soon.</p>
  {% endif %}

</div>

