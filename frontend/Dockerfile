FROM nginx:1.23.3
# Sao chép file cấu hình của Nginx vào container
COPY ./default.conf /etc/nginx/conf.d/default.conf

# Copy tập tin tĩnh (static files) của ReactJS vào thư mục của Nginx
COPY ./build /usr/share/nginx/html

# Expose cổng 80
EXPOSE 80

# Khởi động Nginx khi container chạy
CMD ["nginx", "-g", "daemon off;"]