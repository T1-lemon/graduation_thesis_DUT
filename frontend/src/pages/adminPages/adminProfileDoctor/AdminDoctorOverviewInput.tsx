import React from 'react'

type Props = {
  title?: string
  onChangeTitle: (event: React.ChangeEvent<HTMLInputElement>) => void 
}

function AdminDoctorOverviewInput(props: Props) {
  const { title, onChangeTitle } = props
  return (
    <div className='overview__div--title'>
      <p className='overview__p--title'>Title</p>
      <input
        type='text'
        className='overview__input--title'
        defaultValue={title}
        onChange={onChangeTitle}
      />
    </div>
  )
}

export default AdminDoctorOverviewInput
