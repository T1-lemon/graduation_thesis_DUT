type Props = {
  title: String
  detail: String
}

export default function AuthBoxHeader(props: Props) {
  const { title, detail } = props
  return (
    <div className='auth__box--header'>
      <p className='auth__title'>{title}</p>
      <span className='auth__detail'>{detail}</span>
    </div>
  )
}
