---
layout: default
permalink: /blog/
title: blog
nav: true
nav_order: 1
pagination:
  enabled: true
  collection: posts
  permalink: /page/:num/
  per_page: 5
  sort_field: date
  sort_reverse: true
---

<div class="post">

{% assign postlist = page.pagination.enabled and paginator.posts or site.posts %}

{% if postlist and postlist.size > 0 %}
<ul class="post-list">
{% for post in postlist %}
<li>
<h3>
<a class="post-title" href="{{ post.url | relative_url }}">{{ post.title }}</a>
</h3>
{% if post.description %}<p>{{ post.description }}</p>{% endif %}
<p class="post-meta">{{ post.date | date: '%B %d, %Y' }}</p>
</li>
{% endfor %}
</ul>

    {% if page.pagination.enabled %}
      {% include pagination.liquid %}
    {% endif %}

{% else %}
<p>No posts yet. Check back soon.</p>
{% endif %}

</div>
